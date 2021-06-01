import sys

sys.argv += [
    "--output_dir",
    "",
    "--mode",
    "test",
    "--input_dir",
    "./data/maps/val",
    "--checkpoint",
    "./models/maps_BtoA",
]
from pix2pix import *


# pix2pix #
def load_example(input_image_file=None, file_type="jpg"):
    if a.input_dir is None or not os.path.exists(a.input_dir):
        raise Exception("input_dir does not exist")
    if input_image_file:
        if file_type == "jpg":
            decode = tf.image.decode_jpeg
        elif file_type == "png":
            decode = tf.image.decode_png
        else:
            raise Exception("Invalid file type")
    else:
        input_paths = glob.glob(os.path.join(a.input_dir, "*.jpg"))
        decode = tf.image.decode_jpeg
        if len(input_paths) == 0:
            input_paths = glob.glob(os.path.join(a.input_dir, "*.png"))
            decode = tf.image.decode_png

    if input_image_file:
        contents = input_image_file.read()
    else:
        if len(input_paths) == 0:
            raise Exception("input_dir contains no image files")

        input_path = [random.choice(input_paths)]

        def get_name(path):
            name, _ = os.path.splitext(os.path.basename(path))
            return name

        with tf.name_scope("load_images"):
            path_queue = tf.train.string_input_producer(
                input_path, shuffle=a.mode == "train"
            )
            reader = tf.WholeFileReader()
            paths, contents = reader.read(path_queue)

    raw_input = decode(contents)
    raw_input = tf.image.convert_image_dtype(raw_input, dtype=tf.float32)

    assertion = tf.assert_equal(
        tf.shape(raw_input)[2], 3, message="image does not have 3 channels"
    )
    with tf.control_dependencies([assertion]):
        raw_input = tf.identity(raw_input)

    raw_input.set_shape([None, None, 3])

    if input_image_file:
        # crop image to be square
        shape = tf.shape(raw_input)
        side_len = tf.math.minimum(shape[0], shape[1])
        crop_len_height = (shape[0] - side_len) // 2
        crop_len_width = (shape[1] - side_len) // 2
        a_images = preprocess(
            raw_input[
                crop_len_height + 1 : crop_len_height + side_len + 1,
                crop_len_width : crop_len_width + side_len,
                :,
            ]
        )
        b_images = a_images
    else:
        # break apart image pair and move to range [-1, 1]
        width = tf.shape(raw_input)[1]  # [height, width, channels]
        a_images = preprocess(raw_input[:, : width // 2, :])
        b_images = preprocess(raw_input[:, width // 2 :, :])

    inputs, targets = [b_images, a_images]

    # synchronize seed for image operations so that we do the same operations to both
    # input and output images
    seed = random.randint(0, 2 ** 31 - 1)

    def transform(image):
        r = image
        if a.flip:
            r = tf.image.random_flip_left_right(r, seed=seed)

        # area produces a nice downscaling, but does nearest neighbor for upscaling
        # assume we're going to be doing downscaling here
        r = tf.image.resize_images(
            r, [a.scale_size, a.scale_size], method=tf.image.ResizeMethod.AREA
        )

        offset = tf.cast(
            tf.floor(
                tf.random_uniform(
                    [2], 0, a.scale_size - CROP_SIZE + 1, seed=seed
                )
            ),
            dtype=tf.int32,
        )
        if a.scale_size > CROP_SIZE:
            r = tf.image.crop_to_bounding_box(
                r, offset[0], offset[1], CROP_SIZE, CROP_SIZE
            )
        elif a.scale_size < CROP_SIZE:
            raise Exception("scale size cannot be less than crop size")
        return r

    with tf.name_scope("input_images"):
        input_images = transform(inputs)

    with tf.name_scope("target_images"):
        target_images = transform(targets)

    inputs_batch, targets_batch = tf.train.batch(
        [input_images, target_images], batch_size=a.batch_size
    )
    steps_per_epoch = 1

    return Examples(
        paths=[],
        inputs=inputs_batch,
        targets=targets_batch,
        count=1,
        steps_per_epoch=steps_per_epoch,
    )


if a.seed is None:
    a.seed = random.randint(0, 2 ** 31 - 1)

tf.set_random_seed(a.seed)
np.random.seed(a.seed)
random.seed(a.seed)

if a.mode == "test" or a.mode == "export":
    if a.checkpoint is None:
        raise Exception("checkpoint required for test mode")

    # load some options from the checkpoint
    options = {"which_direction", "ngf", "ndf", "lab_colorization"}
    with open(os.path.join(a.checkpoint, "options.json")) as f:
        for key, val in json.loads(f.read()).items():
            if key in options:
                print("loaded", key, "=", val)
                setattr(a, key, val)
    # disable these features in test mode
    a.scale_size = CROP_SIZE
    a.flip = False


def generate_example(**kwargs):
    examples = load_example(**kwargs)
    print("examples count = %d" % examples.count)

    # inputs and targets are [batch_size, height, width, channels]
    model = create_model(examples.inputs, examples.targets)

    inputs = deprocess(examples.inputs)
    targets = deprocess(examples.targets)
    outputs = deprocess(model.outputs)

    def convert(image):
        if a.aspect_ratio != 1.0:
            # upscale to correct aspect ratio
            size = [CROP_SIZE, int(round(CROP_SIZE * a.aspect_ratio))]
            image = tf.image.resize_images(
                image, size=size, method=tf.image.ResizeMethod.BICUBIC
            )

        return tf.image.convert_image_dtype(
            image, dtype=tf.uint8, saturate=True
        )

    # reverse any processing on images so they can be written to disk or displayed to user
    with tf.name_scope("convert_inputs"):
        converted_inputs = convert(inputs)

    with tf.name_scope("convert_targets"):
        converted_targets = convert(targets)

    with tf.name_scope("convert_outputs"):
        converted_outputs = convert(outputs)

    final_images = {
        "input": converted_inputs,
        "output": converted_outputs,
        "target": converted_targets,
    }

    saver = tf.train.Saver(max_to_keep=1)

    sv = tf.train.Supervisor(saver=None)
    with sv.managed_session() as sess:
        if a.checkpoint is not None:
            print("loading model from checkpoint")
            checkpoint = tf.train.latest_checkpoint(a.checkpoint)
            saver.restore(sess, checkpoint)

        results = sess.run(final_images)
    return results
