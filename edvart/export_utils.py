import base64
import inspect
import os


def embed_image_base64(image_path: str, mime: str = "image/png") -> str:
    """
    Loads content of an image and embeds it as base64 data URL into the template.
    Intended to be used as a Jinja filter.

    Example Jinja filter usage (CSS):
    ```
    #notebook-container {
            background-image: url('{{ 'background.png' | embed_image_base64('image/png') }}');
    ```

    Parameters
    ----------
    image_path : str
        Relative path from the current template to the image.
    mime : str
        Mime type of the image.

    Returns
    -------
    str
        Data URL with embedded image and mime type specified.
    """
    # Look up directory where currently executed template is located
    # Jinja's @environmentfilter or @contextfilter does not seem to provide
    # any information about the path of the template.
    template_dir = os.path.dirname(inspect.getfile(inspect.currentframe().f_back))
    with open(os.path.join(template_dir, image_path), "rb") as img:
        return f"data:{mime};base64," + str(base64.b64encode(img.read()).decode("utf-8"))
