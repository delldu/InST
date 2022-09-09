import image_shape_style

# image_shape_style.image_client("TAI", "images/*.png", "output")
# image_shape_style.image_server("TAI")

image_shape_style.image_predict("images/source_rgb/*.png", "images/source/*.png", "images/target/*.png", "output")
