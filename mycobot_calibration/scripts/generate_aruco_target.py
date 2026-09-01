#!/usr/bin/env python3
"""Generate the ArUco calibration target assets used by the URDF.

Writes two files into mycobot_description/meshes/aruco_marker/:

  aruco_<dict>_id<id>.png  -- the marker bitmap plus a white quiet zone
  aruco_<dict>_id<id>.dae  -- a flat textured plane sized to the plate

The plane lies in its own XY plane with the surface normal along +Z, so the
URDF can aim the marker simply by rotating the mesh. Marker frame axes follow
the OpenCV convention returned by estimatePoseSingleMarkers: +X right along
the top edge, +Y up, +Z out of the marker face.

A white quiet zone is mandatory -- ArUco cannot segment a marker whose black
border touches the edge of the geometry.

Run inside the container:
    python3 generate_aruco_target.py

:author: hand-eye calibration support for myCobot 280
"""

import argparse
import os
import sys

import cv2

# Runnable straight out of the source tree, before the package is built, since
# the assets it writes are inputs to the URDF rather than outputs of a build.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from mycobot_calibration.aruco_compat import (   # noqa: E402
    create_charuco_board,
    generate_board_image,
    generate_marker_image,
    get_dictionary,
)

# The DICT_* enums are the one part of the aruco module that survived the 4.7
# API break unchanged; everything else goes through aruco_compat.
DICTIONARIES = {
    'DICT_4X4_50': cv2.aruco.DICT_4X4_50,
    'DICT_4X4_250': cv2.aruco.DICT_4X4_250,
    'DICT_5X5_250': cv2.aruco.DICT_5X5_250,
    'DICT_6X6_250': cv2.aruco.DICT_6X6_250,
    'DICT_7X7_250': cv2.aruco.DICT_7X7_250,
    'DICT_ARUCO_ORIGINAL': cv2.aruco.DICT_ARUCO_ORIGINAL,
}

COLLADA_TEMPLATE = """<?xml version="1.0" encoding="UTF-8"?>
<COLLADA xmlns="http://www.collada.org/2005/11/COLLADASchema" version="1.4.1">
  <asset>
    <contributor>
      <author>mycobot_calibration/generate_aruco_target.py</author>
    </contributor>
    <up_axis>Z_UP</up_axis>
    <unit meter="1.0" name="meter"/>
  </asset>

  <library_images>
    <image id="marker_tex" name="marker_tex">
      <init_from>{texture}</init_from>
    </image>
  </library_images>

  <library_effects>
    <effect id="marker_fx">
      <profile_COMMON>
        <newparam sid="marker_surface">
          <surface type="2D">
            <init_from>marker_tex</init_from>
            <format>R8G8B8</format>
          </surface>
        </newparam>
        <newparam sid="marker_sampler">
          <sampler2D>
            <source>marker_surface</source>
            <minfilter>LINEAR_MIPMAP_LINEAR</minfilter>
            <magfilter>LINEAR</magfilter>
          </sampler2D>
        </newparam>
        <technique sid="common">
          <lambert>
            <emission>
              <color>0 0 0 1</color>
            </emission>
            <!-- Ambient is textured as well as diffuse. With a textured ambient
                 term the print stays black-on-white under the world's high
                 scene ambient; a flat grey ambient would lift the black squares
                 and cost the detector its contrast. -->
            <ambient>
              <texture texture="marker_sampler" texcoord="UVSET0"/>
            </ambient>
            <diffuse>
              <texture texture="marker_sampler" texcoord="UVSET0"/>
            </diffuse>
            <!-- Matte: a specular highlight sliding across the plate would
                 blow out white squares and break corner refinement. -->
            <reflective>
              <color>0 0 0 1</color>
            </reflective>
            <transparency>
              <float>1.0</float>
            </transparency>
          </lambert>
        </technique>
      </profile_COMMON>
    </effect>

    <!-- Blank back of the plate. A single-sided plane is invisible from
         behind, which reads as the target vanishing rather than as a plate
         seen from the wrong side. The back is plain white so it can never be
         mistaken for a marker: a mirrored ArUco pattern is not a valid code,
         but giving the back its own material removes the question entirely. -->
    <effect id="blank_fx">
      <profile_COMMON>
        <technique sid="common">
          <lambert>
            <emission><color>0 0 0 1</color></emission>
            <ambient><color>0.85 0.85 0.85 1</color></ambient>
            <diffuse><color>0.85 0.85 0.85 1</color></diffuse>
            <reflective><color>0 0 0 1</color></reflective>
          </lambert>
        </technique>
      </profile_COMMON>
    </effect>
  </library_effects>

  <library_materials>
    <material id="marker_mat" name="marker_mat">
      <instance_effect url="#marker_fx"/>
    </material>
    <material id="blank_mat" name="blank_mat">
      <instance_effect url="#blank_fx"/>
    </material>
  </library_materials>

  <library_geometries>
    <geometry id="plate" name="plate">
      <mesh>
        <source id="plate_pos">
          <float_array id="plate_pos_arr" count="24">{positions}</float_array>
          <technique_common>
            <accessor source="#plate_pos_arr" count="8" stride="3">
              <param name="X" type="float"/>
              <param name="Y" type="float"/>
              <param name="Z" type="float"/>
            </accessor>
          </technique_common>
        </source>
        <source id="plate_norm">
          <float_array id="plate_norm_arr" count="6">0 0 1 0 0 -1</float_array>
          <technique_common>
            <accessor source="#plate_norm_arr" count="2" stride="3">
              <param name="X" type="float"/>
              <param name="Y" type="float"/>
              <param name="Z" type="float"/>
            </accessor>
          </technique_common>
        </source>
        <source id="plate_uv">
          <float_array id="plate_uv_arr" count="8">0 0 1 0 1 1 0 1</float_array>
          <technique_common>
            <accessor source="#plate_uv_arr" count="4" stride="2">
              <param name="S" type="float"/>
              <param name="T" type="float"/>
            </accessor>
          </technique_common>
        </source>
        <vertices id="plate_vtx">
          <input semantic="POSITION" source="#plate_pos"/>
        </vertices>
        <triangles count="2" material="marker_mat">
          <input semantic="VERTEX" source="#plate_vtx" offset="0"/>
          <input semantic="NORMAL" source="#plate_norm" offset="1"/>
          <input semantic="TEXCOORD" source="#plate_uv" offset="2" set="0"/>
          <p>0 0 0  1 0 1  2 0 2   0 0 0  2 0 2  3 0 3</p>
        </triangles>
        <!-- Reversed winding, and offset behind the printed face rather than
             coplanar with it, which would z-fight. -->
        <triangles count="2" material="blank_mat">
          <input semantic="VERTEX" source="#plate_vtx" offset="0"/>
          <input semantic="NORMAL" source="#plate_norm" offset="1"/>
          <input semantic="TEXCOORD" source="#plate_uv" offset="2" set="0"/>
          <p>4 1 0  6 1 2  5 1 1   4 1 0  7 1 3  6 1 2</p>
        </triangles>
      </mesh>
    </geometry>
  </library_geometries>

  <library_visual_scenes>
    <visual_scene id="scene" name="scene">
      <node id="plate_node" name="plate_node" type="NODE">
        <instance_geometry url="#plate">
          <bind_material>
            <technique_common>
              <instance_material symbol="marker_mat" target="#marker_mat">
                <bind_vertex_input semantic="UVSET0" input_semantic="TEXCOORD"
                                   input_set="0"/>
              </instance_material>
              <instance_material symbol="blank_mat" target="#blank_mat"/>
            </technique_common>
          </bind_material>
        </instance_geometry>
      </node>
    </visual_scene>
  </library_visual_scenes>

  <scene>
    <instance_visual_scene url="#scene"/>
  </scene>
</COLLADA>
"""


def build_texture(dictionary_name, marker_id, marker_px, border_ratio):
    """Render the marker bitmap surrounded by a white quiet zone.

    Returns the image and the border width in pixels.
    """
    dictionary = get_dictionary(DICTIONARIES[dictionary_name])
    marker = generate_marker_image(dictionary, marker_id, marker_px)

    border_px = int(round(marker_px * border_ratio))
    image = cv2.copyMakeBorder(
        marker, border_px, border_px, border_px, border_px,
        cv2.BORDER_CONSTANT, value=255)
    return image, border_px


BACK_FACE_OFFSET = 0.0006   # metres behind the printed face


def build_collada(texture_name, width, height=None):
    """Return COLLADA text for a width x height plate facing +Z.

    Vertex order matches the UV array in the template: (0,0), (1,0), (1,1),
    (0,1).

    The printed face sits exactly at z = 0, not offset by half a plate
    thickness. That matters: the link origin is what TF publishes as
    aruco_marker_link, and OpenCV reports the target pose at the printed
    surface, so any offset between them would show up as a fixed bias when
    the detector is scored against ground truth.
    """
    if height is None:
        height = width
    half_w, half_h = width / 2.0, height / 2.0
    front = [
        (-half_w, -half_h, 0.0),
        (half_w, -half_h, 0.0),
        (half_w, half_h, 0.0),
        (-half_w, half_h, 0.0),
    ]
    back = [(x, y, -BACK_FACE_OFFSET) for x, y, _ in front]
    positions = ' '.join(f'{v:.6f}' for corner in front + back for v in corner)
    return COLLADA_TEMPLATE.format(texture=texture_name, positions=positions)


def build_charuco_texture(dictionary_name, squares_x, squares_y,
                          square_length, marker_length, margin_ratio,
                          px_per_square):
    """Render a ChArUco board with a white quiet zone.

    Returns (image, plate_width, plate_height).

    A ChArUco board beats a single marker on the error that dominates here.
    ArUco localises the *outer edge* of a black square, and an anti-aliased
    edge biases the refined corner inward by a fraction of a pixel -- a
    systematic error that no amount of averaging removes. A chessboard corner
    is a saddle point between two black and two white quadrants, so blur is
    symmetric about it and the bias largely cancels. On top of that the pose
    is fitted from every interior corner rather than four, and it still works
    when only part of the board is visible, which matters when the sweep
    deliberately drives the target to the edge of the frame.
    """
    dictionary = get_dictionary(DICTIONARIES[dictionary_name])
    board = create_charuco_board(squares_x, squares_y, square_length,
                                 marker_length, dictionary)

    margin_px = int(round(px_per_square * margin_ratio))
    image = generate_board_image(
        board,
        squares_x * px_per_square + 2 * margin_px,
        squares_y * px_per_square + 2 * margin_px,
        margin_px, 1)

    margin_length = margin_ratio * square_length
    return (image,
            squares_x * square_length + 2 * margin_length,
            squares_y * square_length + 2 * margin_length)


def main():
    default_out = os.path.normpath(os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        '..', '..', 'mycobot_description', 'meshes', 'aruco_marker'))

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--type', default='aruco', choices=['aruco', 'charuco'],
                        dest='target_type',
                        help='Single ArUco marker, or a ChArUco board.')
    parser.add_argument('--dictionary', default='DICT_6X6_250',
                        choices=sorted(DICTIONARIES))
    parser.add_argument('--id', type=int, default=0, dest='marker_id')

    # ChArUco geometry. Defaults give a 5x5 board on a ~121 mm plate with 16
    # interior corners. A smaller dictionary than the single-marker default is
    # deliberate: the embedded markers are only 0.75 of a square, so at the far
    # end of the sweep they span barely a dozen pixels and 4x4 codes stay
    # decodable where 6x6 would not. Only a few need to decode -- they identify
    # the board, and the corners are then interpolated from the chessboard.
    parser.add_argument('--squares-x', type=int, default=5)
    parser.add_argument('--squares-y', type=int, default=5)
    parser.add_argument('--square-length', type=float, default=0.022)
    parser.add_argument('--charuco-marker-length', type=float, default=0.0165)
    parser.add_argument('--margin-ratio', type=float, default=0.25,
                        help='White quiet zone as a fraction of one square.')
    parser.add_argument('--px-per-square', type=int, default=300)
    parser.add_argument('--marker-size', type=float, default=0.04,
                        help='Physical edge length of the black marker square '
                             'in metres. This is the value the detector must '
                             'be told; the plate itself is larger.')
    parser.add_argument('--border-ratio', type=float, default=0.25,
                        help='White quiet zone as a fraction of the marker '
                             'edge, applied on every side.')
    parser.add_argument('--marker-px', type=int, default=1200,
                        help='Texture resolution of the marker square. Kept '
                             'high so the squares stay crisp when the '
                             'renderer mipmaps the plate down.')
    parser.add_argument('--output-dir', default=default_out)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if args.target_type == 'charuco':
        stem = (f'charuco_{args.dictionary.lower()}_'
                f'{args.squares_x}x{args.squares_y}')
        image, plate_width, plate_height = build_charuco_texture(
            args.dictionary, args.squares_x, args.squares_y,
            args.square_length, args.charuco_marker_length,
            args.margin_ratio, args.px_per_square)
    else:
        stem = f'aruco_{args.dictionary.lower()}_id{args.marker_id}'
        image, border_px = build_texture(
            args.dictionary, args.marker_id, args.marker_px, args.border_ratio)
        plate_width = plate_height = (
            args.marker_size * (image.shape[0] / float(args.marker_px)))

    texture_name = f'{stem}.png'
    mesh_name = f'{stem}.dae'

    cv2.imwrite(os.path.join(args.output_dir, texture_name), image)
    with open(os.path.join(args.output_dir, mesh_name), 'w') as handle:
        handle.write(build_collada(texture_name, plate_width, plate_height))

    print(f'target type : {args.target_type}')
    print(f'dictionary  : {args.dictionary}')
    if args.target_type == 'charuco':
        interior = (args.squares_x - 1) * (args.squares_y - 1)
        print(f'board       : {args.squares_x}x{args.squares_y} squares, '
              f'{interior} interior corners  <- the detector fits all of these')
        print(f'square      : {args.square_length:.4f} m')
        print(f'marker      : {args.charuco_marker_length:.4f} m')
        print('give the detector: squares_x, squares_y, square_length, '
              'charuco_marker_length')
    else:
        print(f'marker id   : {args.marker_id}')
        print(f'marker size : {args.marker_size:.4f} m  '
              '<- give this to the detector')
        print(f'quiet zone  : {border_px} px')
    print(f'plate size  : {plate_width:.4f} x {plate_height:.4f} m  '
          '<- geometry size used by the URDF')
    print(f'texture     : {image.shape[1]}x{image.shape[0]} px')
    print(f'written to  : {args.output_dir}')


if __name__ == '__main__':
    main()
