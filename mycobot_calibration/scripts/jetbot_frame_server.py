#!/usr/bin/env python3
"""Ship D435 frames off the JetBot to whatever host runs the detector.

The camera is bolted to the JetBot, which is a Jetson Nano on Ubuntu 18.04
running ROS 2 Eloquent -- five distros from the Humble stack that owns the arm,
on a different subnet, so the two ROS graphs cannot usefully be joined. This
script sidesteps the problem entirely: it is a plain TCP frame pusher with no
ROS dependency at all, and the detector runs on the arm's host instead.

WHY V4L2 AND NOT LIBREALSENSE
-----------------------------
The D435 presents its streams as ordinary UVC video nodes, so the colour stream
can be read with cv2.VideoCapture and nothing else. That matters here: this Nano
has librealsense 2.54 but no pyrealsense2, and no aarch64 wheel exists, so the
Python binding would have to be built from source. It is also mutually exclusive
with capture -- rs-enumerate-devices claims the device, so calling it while this
script is streaming fails. Intrinsics are therefore read ONCE, by hand, and
passed to the receiving node as parameters; they never change for a given
camera and resolution.

    rs-enumerate-devices -c | grep -A 9 'Intrinsic of "Color" / 848x480'

Note which node is which: on this unit /dev/video2 carries the depth/IR modes
(it advertises 848x100, a depth-only mode) and /dev/video3 is Colour.

WHY LOSSLESS -- AND WHY THAT TURNED OUT TO MATTER LESS THAN EXPECTED
--------------------------------------------------------------------
The dominant error in this calibration is sub-pixel corner bias -- roughly half
a pixel per marker edge, which reads as a proportional over-estimate of every
distance. JPEG ringing around the marker's high-contrast border adds exactly
that kind of bias, and it would be invisible in the result: the calibration
would simply come out consistently wrong. Hence the lossless default.

That argument is sound but the magnitude was never checked, and this comment
used to end 'use --format jpeg only for a quick liveness check, never for a
sweep'. MEASURED 2026-09-03, by re-encoding 89 captured lossless frames and
re-detecting -- identical pixels, only the codec differing, so nothing physical
can contaminate the comparison:

    codec              mean span      distance bias
    lossless           106.107 px     --
    JPEG q95           106.100 px     +0.019 mm  (+0.009%)
    JPEG q85           106.012 px     +0.289 mm  (+0.130%)
    JPEG q70           106.308 px     +0.163 mm  (+0.074%)

At q95 the bias is 19 micrometres at 0.22 m, about 120x smaller than the arm's
own 2.3 mm RMS execution error. So q95 is fine for a sweep. It is also CHEAPER
than lossless mono on both counts, because PNG compression is the expensive
part: 90 kB/frame and 6.8% Nano CPU for bgr8 JPEG, against 160 kB and 20.7% for
mono8 PNG.

Two caveats before leaning on this. It was measured at one pose, with the
marker spanning 106 px; ringing is a fixed number of pixels, so its relative
cost grows as the marker shrinks, and the far end of a sweep is exactly where
the marker is smallest. And q85 is already 15x worse than q95, so do not drop
the quality to save bandwidth. When in doubt, --format png costs only CPU.

Greyscale, because ArUco only ever looks at the luma channel and mono8 is a
third of the bytes; --colour exists for aiming the camera by eye.

At 848x480 lossless mono is about 100 kB a frame, so 10 Hz costs roughly
8 Mbit/s.

Usage on the JetBot:

    python3 jetbot_frame_server.py --device /dev/video3 --port 5555

Then on the host that runs the arm:

    ros2 run mycobot_calibration jetbot_camera_bridge \\
        --ros-args -p host:=172.22.133.39
"""

import argparse
import json
import socket
import struct
import sys
import time

import cv2

# [payload length][capture time, microseconds since epoch]
FRAME_HEADER = struct.Struct('>IQ')


def open_camera(device, width, height):
    capture = cv2.VideoCapture(device, cv2.CAP_V4L2)
    if not capture.isOpened():
        raise SystemExit('could not open %s' % device)
    # YUYV is what the D435 colour node actually advertises; leaving OpenCV to
    # negotiate lands on MJPG on some builds, which would reintroduce exactly
    # the lossy compression this script exists to avoid.
    capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'YUYV'))
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    actual_w = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if (actual_w, actual_h) != (width, height):
        print('warning: asked for %dx%d, got %dx%d'
              % (width, height, actual_w, actual_h))
    return capture, actual_w, actual_h


def encode(frame, fmt, png_level, jpeg_quality, colour):
    """Encode a frame, in colour or greyscale.

    Greyscale is the default because the detector discards colour as its very
    first act -- aruco_detector_node converts to grey before it looks at
    anything -- so mono8 carries exactly the information ArUco uses at a third
    of the bytes. Over a link this marginal that is not a micro-optimisation:
    colour roughly triples the wire rate.

    Colour is worth having anyway for aiming the camera by eye, which is a human
    task and much easier in colour. Use it for setup, not for a sweep.
    """
    image = frame if colour else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if fmt == 'png':
        ok, buf = cv2.imencode(
            '.png', image, [cv2.IMWRITE_PNG_COMPRESSION, png_level])
    else:
        ok, buf = cv2.imencode(
            '.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
    if not ok:
        return None
    return buf.tobytes()


def serve_client(connection, capture, args, width, height):
    """Stream frames, decoding only the ones actually being sent.

    The loop is built around cv2's grab/retrieve split rather than read(),
    because on a Nano the difference is most of the CPU cost. grab() only
    dequeues a V4L2 buffer; retrieve() is what performs the YUYV conversion. So
    grabbing every frame keeps the driver's queue drained -- which is what makes
    the frame we send always the current one -- while conversion and PNG
    encoding happen at the wire rate instead of the camera rate.

    That matters for correctness as well as load. V4L2 hands OpenCV a small
    queue of buffers, and a reader slower than the camera drains them in order
    and falls steadily further behind. During a sweep that is the one failure
    that would corrupt the data rather than merely slow it down: the collector
    settles the arm, then samples, and a stale queued frame from before the move
    would be attributed to the pose after it.

    An earlier version ran a grabber thread calling read() flat out, converting
    every frame at camera rate and keeping the newest. It was correct but did
    roughly twice the work for nothing, on a machine that has none to spare.
    """
    greeting = json.dumps({
        'width': width,
        'height': height,
        'encoding': 'bgr8' if args.colour else 'mono8',
        'format': args.format,
        'fps': args.fps,
    }).encode('utf-8') + b'\n'
    connection.sendall(greeting)

    period = 1.0 / args.fps if args.fps > 0 else 0.0
    sent = 0
    grabbed = 0
    last_report = time.time()
    next_send = time.time()

    while True:
        # Blocks until the camera has another frame, so this paces the loop at
        # the capture rate without spinning.
        if not capture.grab():
            time.sleep(0.01)
            continue
        grabbed += 1
        stamp_us = int(time.time() * 1e6)

        now = time.time()
        if now < next_send:
            continue
        next_send = max(now, next_send + period) if period else now

        ok, frame = capture.retrieve()
        if not ok:
            continue

        payload = encode(frame, args.format, args.png_level,
                         args.jpeg_quality, args.colour)
        if payload is None:
            continue

        connection.sendall(FRAME_HEADER.pack(len(payload), stamp_us) + payload)
        sent += 1

        if now - last_report >= 10.0:
            print('sent %d (%.1f Hz) of %d grabbed, %d kB last'
                  % (sent, sent / (now - last_report), grabbed,
                     len(payload) // 1024))
            sent = 0
            grabbed = 0
            last_report = now


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', default='/dev/video3',
                        help='V4L2 node for the COLOUR stream (not video2, '
                             'which carries depth/IR on this unit)')
    parser.add_argument('--width', type=int, default=848)
    parser.add_argument('--height', type=int, default=480)
    parser.add_argument('--port', type=int, default=5555)
    parser.add_argument('--bind', default='0.0.0.0')
    parser.add_argument('--fps', type=float, default=10.0,
                        help='wire rate; the camera is read as fast as it will '
                             'go regardless, so this only throttles the network')
    parser.add_argument('--format', choices=['png', 'jpeg'], default='png',
                        help='png is lossless and the only correct choice for '
                             'a calibration sweep')
    parser.add_argument('--png-level', type=int, default=1,
                        help='0-9. Low is the right end here: the Nano encodes '
                             'far faster and the extra bytes are cheaper than '
                             'the CPU')
    parser.add_argument('--jpeg-quality', type=int, default=95)
    parser.add_argument('--colour', '--color', action='store_true',
                        dest='colour',
                        help='send bgr8 instead of mono8. Useful for aiming the '
                             'camera by eye; roughly triples the wire rate and '
                             'buys the detector nothing, so leave it off for a '
                             'sweep.')
    args = parser.parse_args()

    capture, width, height = open_camera(args.device, args.width, args.height)

    # Prove the camera actually delivers before advertising a port, so a dead
    # sensor fails here rather than as a client that connects and then hangs.
    ok = False
    deadline = time.time() + 5.0
    while time.time() < deadline:
        ok, _ = capture.read()
        if ok:
            break
        time.sleep(0.05)
    if not ok:
        capture.release()
        raise SystemExit('camera opened but produced no frames')

    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((args.bind, args.port))
    server.listen(1)
    print('serving %dx%d %s/%s on %s:%d'
          % (width, height, 'bgr8' if args.colour else 'mono8',
             args.format, args.bind, args.port))

    try:
        while True:
            connection, address = server.accept()
            # Nagle would coalesce the header with the next frame and add a
            # round trip of latency for no gain; these are already big writes.
            connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            print('client connected from %s:%d' % address)
            try:
                serve_client(connection, capture, args, width, height)
            except (socket.error, OSError) as exc:
                print('client gone (%s)' % exc)
            finally:
                try:
                    connection.close()
                except OSError:
                    pass
    except KeyboardInterrupt:
        print('\nstopping')
    finally:
        capture.release()
        server.close()
    return 0


if __name__ == '__main__':
    sys.exit(main())
