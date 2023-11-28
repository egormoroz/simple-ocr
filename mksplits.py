import json, glob, os
from PIL import Image
from tqdm import tqdm


def make_split(root):
    data = {}
    for folder in tqdm(sorted(glob.glob(root + '*/'))):
        folder_data = []
        with open(folder + 'labels.txt') as f:
            for entry in f:
                fname, text = entry.strip().split(maxsplit=1)
                impath = folder + fname
                if not os.path.isfile(impath):
                    continue

                w, h = Image.open(impath).size
                folder_data.append({
                    'fname': fname,
                    'text': text,
                    'w': w,
                    'h': h,
                })
        data[folder] = folder_data
    return data


for split_path in sorted(glob.glob('./data/*/')):
    split_name = split_path.rsplit('/')[-2]
    print('**', split_name)
    with open(f'data/{split_name}.json', 'w') as f:
        data = make_split(split_path)
        print(sum(len(fol) for fol in data.values()), 'entries')
        json.dump(data, f)

