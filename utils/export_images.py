# export_examples.py

import os
import numpy as np
from PIL import Image

from utils import get_data


def export_examples(dataset, sample_ids, save_dir="Fotos"):

    os.makedirs(save_dir, exist_ok=True)

    for idx in sample_ids:

        image, mask = dataset[idx]

        # Tensor -> numpy
        img_np = image.permute(1, 2, 0).numpy()
        mask_np = mask.numpy()

        # Aplicar mascara
        masked_img = img_np.copy()
        masked_img[mask_np == 0] = 0

        # Convertir a uint8
        img_uint8 = (img_np * 255).astype(np.uint8)
        mask_uint8 = (mask_np * 255).astype(np.uint8)
        masked_uint8 = (masked_img * 255).astype(np.uint8)

        # Guardar imatge original
        Image.fromarray(img_uint8).save(
            os.path.join(save_dir, f"image_{idx}.png")
        )

        # Guardar mascara
        Image.fromarray(mask_uint8).save(
            os.path.join(save_dir, f"mask_{idx}.png")
        )

        # Guardar mascara aplicada
        Image.fromarray(masked_uint8).save(
            os.path.join(save_dir, f"masked_{idx}.png")
        )

        print(f"Exportada mostra {idx}")


if __name__ == "__main__":

    dataset = get_data(slice=1, train=False)

    # IDs que vols exportar
    sample_ids = [0, 5, 20, 42]

    export_examples(dataset, sample_ids)