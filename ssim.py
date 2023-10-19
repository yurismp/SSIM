pip install scikit-image pandas

import os
from skimage import io, img_as_float
from skimage.metrics import structural_similarity as ssim
import pandas as pd


def compare_with_references(image_path, refs):
    """Compare uma imagem com todas as referências usando SSIM."""
    image = img_as_float(io.imread(image_path, as_gray=True))
    # Redimensiona a imagem para ter as mesmas dimensões da primeira imagem de referência
    image = transform.resize(image, (refs["ref_a"].shape[0], refs["ref_a"].shape[1]))
    
    ssim_values = {}

    for ref_name, ref_image in refs.items():
        s, _ = ssim(image, ref_image, full=True)
        ssim_values[ref_name] = s

    return ssim_values

# Carregue as imagens de referência
ref_folder = "/caminho da pasta"
refs = {
    "ref_a": img_as_float(io.imread(os.path.join(ref_folder, "ref_a.png"), as_gray=True)),
    "ref_b": img_as_float(io.imread(os.path.join(ref_folder, "ref_b.png"), as_gray=True))
}

# Ajuste a segunda imagem de referência para ter as mesmas dimensões da primeira
refs["ref_b"] = transform.resize(refs["ref_b"], (refs["ref_a"].shape[0], refs["ref_a"].shape[1]))

# Liste as imagens na pasta "comps" e calcule o SSIM para cada uma
comp_folder = "/caminho da pasta"
comp_files = [f for f in os.listdir(comp_folder) if os.path.isfile(os.path.join(comp_folder, f)) and not f.startswith('.')]

results = []
for comp_file in comp_files:
    comp_path = os.path.join(comp_folder, comp_file)
    ssims = compare_with_references(comp_path, refs)
    results.append([comp_file, ssims["ref_a"], ssims["ref_b"]])

# Crie um dataframe com os resultados
df = pd.DataFrame(results, columns=["filename", "SSIM_ref_a", "SSIM_ref_b"])
print(df)
