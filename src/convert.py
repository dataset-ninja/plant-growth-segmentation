import supervisely as sly
import os, glob
from dataset_tools.convert import unpack_if_archive
import src.settings as s
from urllib.parse import unquote, urlparse
from supervisely.io.fs import get_file_name_with_ext, get_file_name, file_exists, get_file_ext

from tqdm import tqdm

def download_dataset(teamfiles_dir: str) -> str:
    """Use it for large datasets to convert them on the instance"""
    api = sly.Api.from_env()
    team_id = sly.env.team_id()
    storage_dir = sly.app.get_data_dir()

    if isinstance(s.DOWNLOAD_ORIGINAL_URL, str):
        parsed_url = urlparse(s.DOWNLOAD_ORIGINAL_URL)
        file_name_with_ext = os.path.basename(parsed_url.path)
        file_name_with_ext = unquote(file_name_with_ext)

        sly.logger.info(f"Start unpacking archive '{file_name_with_ext}'...")
        local_path = os.path.join(storage_dir, file_name_with_ext)
        teamfiles_path = os.path.join(teamfiles_dir, file_name_with_ext)

        fsize = api.file.get_directory_size(team_id, teamfiles_dir)
        with tqdm(
            desc=f"Downloading '{file_name_with_ext}' to buffer...",
            total=fsize,
            unit="B",
            unit_scale=True,
        ) as pbar:        
            api.file.download(team_id, teamfiles_path, local_path, progress_cb=pbar)
        dataset_path = unpack_if_archive(local_path)

    if isinstance(s.DOWNLOAD_ORIGINAL_URL, dict):
        for file_name_with_ext, url in s.DOWNLOAD_ORIGINAL_URL.items():
            local_path = os.path.join(storage_dir, file_name_with_ext)
            teamfiles_path = os.path.join(teamfiles_dir, file_name_with_ext)

            if not os.path.exists(get_file_name(local_path)):
                fsize = api.file.get_directory_size(team_id, teamfiles_dir)
                with tqdm(
                    desc=f"Downloading '{file_name_with_ext}' to buffer...",
                    total=fsize,
                    unit="B",
                    unit_scale=True,
                ) as pbar:
                    api.file.download(team_id, teamfiles_path, local_path, progress_cb=pbar)

                sly.logger.info(f"Start unpacking archive '{file_name_with_ext}'...")
                unpack_if_archive(local_path)
            else:
                sly.logger.info(
                    f"Archive '{file_name_with_ext}' was already unpacked to '{os.path.join(storage_dir, get_file_name(file_name_with_ext))}'. Skipping..."
                )

        dataset_path = storage_dir
    return dataset_path
    
def count_files(path, extension):
    count = 0
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith(extension):
                count += 1
    return count
    
def convert_and_upload_supervisely_project(
    api: sly.Api, workspace_id: int, project_name: str
) -> sly.ProjectInfo:
    batch_size = 30

    dataset_path = os.path.join("Plant-Growth-Segmentation","train")
    masks_folder = "SegmentationObject"
    images_ext = ".jpg"
    masks_ext = ".png"
    ds_name = "train"
    group_tag_name = "plant_id"


    def create_ann(image_path):
        labels = []

        rep_tag = sly.Tag(rep_to_tag_meta[image_path.split("/")[-3]])
        id_data = image_path.split("/")[-4]
        group_id = sly.Tag(tag_id, value=id_data)

        mask_path = os.path.join(
            image_path.split("/PNGImages")[0], masks_folder, get_file_name_with_ext(image_path)
        )
        mask_np = sly.imaging.image.read(mask_path)[:, :, 0]
        img_height = mask_np.shape[0]
        img_wight = mask_np.shape[1]
        obj_mask = mask_np == 128
        curr_bitmap = sly.Bitmap(obj_mask)
        curr_label = sly.Label(curr_bitmap, obj_class)
        labels.append(curr_label)

        return sly.Annotation(
            img_size=(img_height, img_wight), labels=labels, img_tags=[group_id, rep_tag]
        )


    obj_class = sly.ObjClass("plant", sly.Bitmap)

    tag_rep_01 = sly.TagMeta("rep_01", sly.TagValueType.NONE)
    tag_rep_02 = sly.TagMeta("rep_02", sly.TagValueType.NONE)
    tag_rep_06 = sly.TagMeta("rep_06", sly.TagValueType.NONE)
    tag_rep_07 = sly.TagMeta("rep_07", sly.TagValueType.NONE)
    rep_to_tag_meta = {
        "rep_01": tag_rep_01,
        "rep_02": tag_rep_02,
        "rep_06": tag_rep_06,
        "rep_07": tag_rep_07,
    }
    tag_id = sly.TagMeta(group_tag_name, sly.TagValueType.ANY_STRING)
    group_tag_meta = sly.TagMeta(group_tag_name, sly.TagValueType.ANY_STRING)

    project = api.project.create(workspace_id, project_name, change_name_if_conflict=True)
    meta = sly.ProjectMeta(
        obj_classes=[obj_class], tag_metas=[tag_rep_01, tag_rep_02, tag_rep_06, tag_rep_07]
    )
    meta = meta.add_tag_meta(group_tag_meta)
    api.project.update_meta(project.id, meta.to_json())
    api.project.images_grouping(id=project.id, enable=True, tag_name=group_tag_name)


    dataset = api.dataset.create(project.id, ds_name, change_name_if_conflict=True)


    all_images_pathes = glob.glob(dataset_path + "/*/*/PNGImages/*.png")

    progress = sly.Progress("Create dataset {}".format(ds_name), len(all_images_pathes))

    for img_pathes_batch in sly.batched(all_images_pathes, batch_size=batch_size):
        img_names_batch = [get_file_name_with_ext(image_path) for image_path in img_pathes_batch]

        img_infos = api.image.upload_paths(dataset.id, img_names_batch, img_pathes_batch)
        img_ids = [im_info.id for im_info in img_infos]

        anns_batch = [create_ann(image_path) for image_path in img_pathes_batch]
        api.annotation.upload_anns(img_ids, anns_batch)

        progress.iters_done_report(len(img_names_batch))

    return project

