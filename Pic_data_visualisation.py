#数塔集可视化，按顺序选择前8张图像进行检查
import os
import matplotlib.pyplot as plt
from PIL import Image, UnidentifiedImageError




#jupyter use
# %matplotlib inline
# %config InlineBackend.figure_format = 'retina"

# original_images = [ ]
# images = []
# texts = []
# plt.figure(figsize=(16, 5))
#
# image_paths= [filename for filename in os.listdir('../cat_dataset/images')][:8]
#
# for i,filename in enumerate(image_paths) :
#     name = os.path.splitext(filename)[0]
#
#     image = Image.open('../cat_dataset/images/'+filename).convert("RGB")
#
#     plt.subplot(2,4,i+1)
#     plt.imshow(image)
#     plt.title(f"{filename}")
#     plt.xticks([])
#     plt.yticks([])
#
# plt.tight_layout()
# plt.show()

def display_images(image_folder_path, number_of_images=8, figsize=(16, 5)):
    """
    Display images in a grid with error handling.

    :param image_folder_path: Path to the folder containing images.
    :param number_of_images: Number of images to display.
    :param figsize: Size of the figure to display.
    """
    # Attempt to list all files in the directory
    try:
        file_list = os.listdir(image_folder_path)
    except FileNotFoundError:
        print(f"The directory {image_folder_path} was not found.")
        return
    except NotADirectoryError:
        print(f"{image_folder_path} is not a directory.")
        return
    except PermissionError:
        print(f"No permission to list the directory {image_folder_path}.")
        return

    # Filter the list to the first number_of_images image files
    image_paths = []
    for filename in file_list:
        if len(image_paths) >= number_of_images:
            break
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')):
            image_paths.append(filename)

    # Check if any images were found
    if not image_paths:
        print(f"No image files found in the directory {image_folder_path}.")
        return

    # Display the images
    plt.figure(figsize=figsize)
    for i, filename in enumerate(image_paths):
        try:
            image_path = os.path.join(image_folder_path, filename)
            image = Image.open(image_path).convert("RGB")
            plt.subplot(2, int(number_of_images / 2), i + 1)#为了更好的布局
            plt.imshow(image)
            plt.title(f"{filename}")
            plt.xticks([])
            plt.yticks([])
        except UnidentifiedImageError:
            print(f"File {filename} is not a recognizable image format.")
        except FileNotFoundError:
            print(f"Image file {filename} not found.")
        except Exception as e:
            print(f"An error occurred while processing file {filename}: {e}")

    plt.tight_layout()
    plt.show()




#display_images('../cat_dataset/images')

display_images('W:/Jack_datasets/COCO/dataset/Jack_generate_cat/COCO/images/val2017/')
