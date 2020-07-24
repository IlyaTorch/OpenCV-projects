from argparse import ArgumentParser


def get_arguments_from_console():
    """Reading command line arguments"""
    arg_parser = ArgumentParser(description="Basic OpenCV projects")
    arg_parser.add_argument("--img", type=str, help="Path of the image instead of default using web-cam")

    return arg_parser.parse_args()
