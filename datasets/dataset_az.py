import os
os.environ['AZURE_ACC_NAME'] = 'm2mpublicblobstorage01'
os.environ['AZURE_CONTAINER'] = 'm2mstorageukpublicblob001'
os.environ['AZURE_SAS_TOKEN'] = "sp=racwdl&st=2025-04-14T19:49:32Z&se=2025-05-09T03:49:32Z&sv=2024-11-04&sr=c&sig=tXXNzjr3PcLsWzUNhU98pDbpZ6kmdN%2FWr41ojapmTh0%3D"



from adlfs import AzureBlobFileSystem
import webdataset as wds
from torch.utils.data import DataLoader
from torchvision import transforms as T
from utils.data_filtering import WebdatasetFilter, AspectRatioFilter, pilimg_from_base64, filter_keys
from PIL import Image
from torchvision import transforms
import torchvision.transforms.functional as TF
from torch.utils.data import default_collate
import pdb

def clean_cap(example):
    example["text"] = example["text"].replace('The image is of ', '')
    example["text"] = example["text"].replace('This image is of ', '')
    example["text"] = example["text"].replace('The image is ', '')
    example["text"] = example["text"].replace('This image is ', '')
    example["text"] = example["text"].replace('The image shows ', '')
    example["text"] = example["text"].replace('This image shows ', '')
    example["text"] = example["text"].replace('The image features ', '')
    example["text"] = example["text"].replace('This image features ', '')
    example["text"] = example["text"].replace('This image depicts ', '')
    example["text"] = example["text"].replace('The image depicts ', '')
    return example

class AzureWebDatasetPipeline:
    """
    A reusable WebDataset pipeline that streams from Azure Blob Storage via HTTPS+SAS.
    """

    def __init__(
        self,
        prefix: str,
        batch_size: int = 32,
        num_workers: int = 4,
        resolution: int=512,
        shuffle_shards: int = 100,
        shuffle_samples: int = 1000,
        min_size: int = 0,
        max_ar: float = None,
        max_pwatermark: float = None,
        center_crop: bool = True,
        clean_cap: bool = False,
        caption_key: str = None,
    ):
        # Azure & pipeline config
        self.account_name = os.getenv("AZURE_ACC_NAME")
        self.container = os.getenv("AZURE_CONTAINER")
        self.sas_token = os.getenv("AZURE_SAS_TOKEN")
        self.prefix = prefix
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle_shards = shuffle_shards
        self.shuffle_samples = shuffle_samples
        

        # Initialize AzureBlobFileSystem and list shards
        
        if isinstance(prefix, str):
            self.fs = AzureBlobFileSystem(account_name=self.account_name, sas_token=self.sas_token)
            paths = self.fs.glob(f"{self.container}/{prefix}/*.tar")
            blob_keys = [p.split("/", 1)[1] for p in paths]
        else:
            blob_keys = prefix

        self.shard_urls = [
            f"https://{self.account_name}.blob.core.windows.net/{self.container}/{key}?{self.sas_token}"
            for key in blob_keys
        ]


        def transform(example):
            # resize image
            image = example["image"]
            if not isinstance(image, Image.Image):
                image = pilimg_from_base64(image)
            image = TF.resize(image, resolution, interpolation=transforms.InterpolationMode.BILINEAR)

            if center_crop:
                image = TF.center_crop(image, output_size=(resolution, resolution))
            else:
                # get crop coordinates and crop image
                c_top, c_left, _, _ = transforms.RandomCrop.get_params(image, output_size=(resolution, resolution))
                image = TF.crop(image, c_top, c_left, resolution, resolution)

            image = TF.to_tensor(image)
            image = TF.normalize(image, [0.5], [0.5])

            example["image"] = image
            if caption_key == None:
                try:
                    example["text"] = example["text"].decode("utf-8")
                except:
                    example["text"] = example["text"]
            else:
                example["text"] = example['json'][caption_key]
            
            if clean_cap:
                example = clean_cap(example)
            # example['meta'] = example['meta']
            return example


        pipeline = [
            wds.SimpleShardList(self.shard_urls),
            wds.shuffle(self.shuffle_shards),
            wds.split_by_worker,
            # wds.tarfile_to_samples(),
            wds.tarfile_to_samples(handler=wds.warn_and_continue),
        ]
        
        load_meta = False
        if not max_ar is None:
            pipeline += [wds.select(AspectRatioFilter(max_ar=max_ar))]
            load_meta = True
        if min_size > 0  or max_pwatermark is not None:
            pipeline += [wds.select(WebdatasetFilter(min_size=min_size, max_pwatermark=max_pwatermark))]
            load_meta = True


        pipeline += [
            wds.shuffle(self.shuffle_samples),
            wds.decode("pil", handler=wds.warn_and_continue),]
        if load_meta:
            pipeline += [
                wds.rename(image="jpg;png;jpeg;webp", text="text;txt;caption", json="json", handler=wds.warn_and_continue),
                wds.map(filter_keys({"image", "text", "json"}))]
        else:
            pipeline += [
                wds.rename(image="jpg;png;jpeg;webp", text="text;txt;caption", handler=wds.warn_and_continue),
                wds.map(filter_keys({"image", "text"}))]
            
        pipeline += [
            wds.map(transform, handler=wds.warn_and_continue),
            wds.to_tuple("image", "text"),
            wds.batched(self.batch_size, partial=False, collation_fn=default_collate)
        ]

        self.pipeline = wds.DataPipeline(*pipeline)

    def get_dataloader(self):
        # pipeline = self.build_pipeline()
        loader = DataLoader(
            self.pipeline,
            batch_size=None,
            num_workers=self.num_workers,
            # prefetch_factor=2,
        )
        return loader

if __name__ == '__main__':
    import numpy as np

    SHARDS_URI = [f"vision_datasets/laion-coco/{i:05d}.tar" for i in range(64128)]
    np.random.shuffle(SHARDS_URI)

    prefix = 'vision_datasets/syn_data_wbs'

    dataset = AzureWebDatasetPipeline(
            prefix=prefix,
            batch_size=512,
            num_workers=4,
            resolution=512, 
            shuffle_shards = 10,
            shuffle_samples = 1000,
            min_size=0,
            max_ar=None,
            max_pwatermark=None,
            center_crop=True,
            # caption_key='top_caption',
        )

    dataloader = dataset.get_dataloader()
    for images, captions in dataloader:
        print(images.shape)
        print(captions)
        break 