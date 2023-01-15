from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import ast
import random
import clip


CATEGORIES = {
    'dog': 0,
    'elephant': 1,
    'giraffe': 2,
    'guitar': 3,
    'horse': 4,
    'house': 5,
    'person': 6,
}

# DG: For Domain Generalization
SOURCE_DOMAINS={
    'photo': ['art_painting', 'cartoon', 'sketch'],
    'sketch': ['art_painting', 'cartoon', 'photo'],
    'cartoon': ['art_painting', 'sketch', 'photo'],
    'art_painting': ['photo', 'cartoon', 'sketch'],
}

DESCR_TITLES = [
    "Details ",
    "Edges ",
    "Saturation ",
    "Shades ",
    "Background ",
    "Instances ",
    "Text ",
    "Texture ",
    "Perspective "
]


class PACSDatasetBaseline(Dataset):
    def __init__(self, examples, transform):
        self.examples = examples
        self.transform = transform

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        img_path, y = self.examples[index]
        x = self.transform(Image.open(img_path).convert('RGB'))
        return x, y


class PACSDatasetDisentangle(Dataset):
    def __init__(self, src_examples, tgt_examples, transform):
        self.src_examples = src_examples
        self.tgt_examples = tgt_examples
        self.transform = transform

    def __len__(self):
        return min(len(self.src_examples), len(self.tgt_examples))

    def __getitem__(self, index):
        src_img_path, category_label = self.src_examples[index % self.__len__()]
        src_img = self.transform(Image.open(src_img_path).convert('RGB'))

        tgt_img_path, tgt_category_label = self.tgt_examples[index % self.__len__()]
        tgt_img = self.transform(Image.open(tgt_img_path).convert('RGB'))

        return src_img, category_label, tgt_img, tgt_category_label


class PACSDatasetClipDisentangle(Dataset):
    def __init__(self, src_examples, tgt_examples, transform, data_path):
        self.src_examples = src_examples
        self.tgt_examples = tgt_examples
        self.transform = transform
        self.description_dict = {}

        with open(f'{data_path}/clip_descriptions.txt') as text_file:
            text_data = text_file.read()

        list_dict = ast.literal_eval(text_data)

        for d in list_dict:
            key = f'{data_path}/kfold/{d["image_name"]}'
            item = d['descriptions']
            self.description_dict[key] = item

    def __len__(self):
        return min(len(self.src_examples), len(self.tgt_examples))

    def __getitem__(self, index):
        src_img_path, category_label = self.src_examples[index % self.__len__()]
        src_img = self.transform(Image.open(src_img_path).convert('RGB'))
        src_description = self.description_dict[src_img_path] if src_img_path in self.description_dict else "source"
        src_description = " ".join(src_description).replace(',', '').replace('.', '').replace('-', '')
        tokenized_src_desc = clip.tokenize(src_description).squeeze()

        tgt_img_path, tgt_category_label = self.tgt_examples[index % self.__len__()]
        tgt_img = self.transform(Image.open(tgt_img_path).convert('RGB'))
        tgt_description = self.description_dict[tgt_img_path] if tgt_img_path in self.description_dict else "target"
        tgt_description = " ".join(tgt_description).replace(',', '').replace('.', '').replace('-', '')
        tokenized_tgt_desc = clip.tokenize(tgt_description).squeeze()

        return src_img, category_label, tokenized_src_desc, tgt_img, tgt_category_label, tokenized_tgt_desc


# DG: DOMAIN GENERALIZATION
class PACSDatasetDG(Dataset):
    def __init__(self, examples, transform):
        self.examples = examples
        self.transform = transform

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        img_path, category_label, domain_label = self.examples[index]
        img = self.transform(Image.open(img_path).convert('RGB'))
        return img, category_label, domain_label

# DG
class PACSDatasetDG_Clip(Dataset):
    def __init__(self, examples, transform, data_path):
        self.examples = examples
        self.transform = transform
        self.description_dict = {}

        with open(f'{data_path}/clip_descriptions.txt') as text_file:
            text_data = text_file.read()

        list_dict = ast.literal_eval(text_data)

        for d in list_dict:
            key = f'{data_path}/kfold/{d["image_name"]}'
            item = d['descriptions']
            self.description_dict[key] = item

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        img_path, category_label, domain_label = self.examples[index]
        img = self.transform(Image.open(img_path).convert('RGB'))
        description = self.description_dict[img_path] if img_path in self.description_dict else str(domain_label)
        description = " ".join(description).replace(',', '').replace('.', '').replace('-', '')
        tokenized_text = clip.tokenize(description).squeeze()
        return img, category_label, domain_label, tokenized_text



def read_lines(data_path, domain_name):
    examples = {}
    with open(f'{data_path}/{domain_name}.txt') as f:
        lines = f.readlines()

    for line in lines: 
        line = line.strip().split()[0].split('/')
        category_name = line[3]
        category_idx = CATEGORIES[category_name]
        image_name = line[4]
        image_path = f'{data_path}/kfold/{domain_name}/{category_name}/{image_name}'
        if category_idx not in examples.keys():
            examples[category_idx] = [image_path]
        else:
            examples[category_idx].append(image_path)
    return examples

def build_splits_baseline(opt):

    if opt['domain_generalization']:
        return build_splits_domain_generalization(opt)
    else:
        source_domain = 'art_painting'
        target_domain = opt['target_domain']

        source_examples = read_lines(opt['data_path'], source_domain)
        target_examples = read_lines(opt['data_path'], target_domain)

        # Compute ratios of examples for each category
        source_category_ratios = {category_idx: len(examples_list) for category_idx, examples_list in source_examples.items()}
        source_total_examples = sum(source_category_ratios.values())
        source_category_ratios = {category_idx: c / source_total_examples for category_idx, c in source_category_ratios.items()}

        # Build splits - we train only on the source domain (Art Painting)
        val_split_length = source_total_examples * 0.2 # 20% of the training split used for validation

        train_examples = []
        val_examples = []
        test_examples = []

        for category_idx, examples_list in source_examples.items():
            split_idx = round(source_category_ratios[category_idx] * val_split_length)
            for i, example in enumerate(examples_list):
                if i > split_idx:
                    train_examples.append([example, category_idx])  # each pair is [path_to_img, class_label]
                else:
                    val_examples.append([example, category_idx])  # each pair is [path_to_img, class_label]

        for category_idx, examples_list in target_examples.items():
            for example in examples_list:
                test_examples.append([example, category_idx])  # each pair is [path_to_img, class_label]

        # Transforms
        normalize = T.Normalize([0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # ResNet18 - ImageNet Normalization

        train_transform = T.Compose([
            T.Resize(256),
            T.RandAugment(3, 15),
            T.CenterCrop(224),
            T.ToTensor(),
            normalize
        ])

        eval_transform = T.Compose([
            T.Resize(256),
            T.CenterCrop(224),
            T.ToTensor(),
            normalize
        ])

        # Dataloaders
        train_loader = DataLoader(PACSDatasetBaseline(train_examples, train_transform), batch_size=opt['batch_size'], num_workers=opt['num_workers'], shuffle=True)
        val_loader = DataLoader(PACSDatasetBaseline(val_examples, eval_transform), batch_size=opt['batch_size'], num_workers=opt['num_workers'], shuffle=False)
        test_loader = DataLoader(PACSDatasetBaseline(test_examples, eval_transform), batch_size=opt['batch_size'], num_workers=opt['num_workers'], shuffle=False)

        return train_loader, val_loader, test_loader

def build_splits_domain_disentangle(opt):

    if opt['domain_generalization']:
        return build_splits_domain_generalization(opt)
    else:
        source_domain = 'art_painting'
        target_domain = opt['target_domain']

        source_examples = read_lines(opt['data_path'], source_domain)
        target_examples = read_lines(opt['data_path'], target_domain)


        # SOURCE DOMAIN
        # Compute ratios of examples for each category
        source_category_ratios = {category_idx: len(examples_list) for category_idx, examples_list in source_examples.items()}
        source_total_examples = sum(source_category_ratios.values())
        source_category_ratios = {category_idx: c / source_total_examples for category_idx, c in source_category_ratios.items()}

        val_split_length = source_total_examples * 0.4  # 20% of the training split used for validation and 20% for test

        # Build splits - we train only on the source domain (Art Painting)
        train_source_examples = []
        val_source_examples = []
        test_source_examples = []

        for category_idx, examples_list in source_examples.items():
            split_idx1 = round(source_category_ratios[category_idx] * val_split_length)
            split_idx2 = split_idx1 // 2
            for i, example in enumerate(examples_list):
                if i > split_idx1:
                    train_source_examples.append([example, category_idx])  # each pair is [path_to_img, class_label]
                elif split_idx2 <= i < split_idx1:
                    val_source_examples.append([example, category_idx])    # each pair is [path_to_img, class_label]
                else:
                    test_source_examples.append([example, category_idx])   # each pair is [path_to_img, class_label]

        # TARGET DOMAIN
        # Compute ratios of examples for each category
        target_category_ratios = {category_idx: len(examples_list) for category_idx, examples_list in target_examples.items()}
        target_total_examples = sum(target_category_ratios.values())
        target_category_ratios = {category_idx: c / target_total_examples for category_idx, c in target_category_ratios.items()}

        val_split_length = target_total_examples * 0.4  # 20% of the training split used for validation and 20% for test

        # Build splits - we train only on the source domain (Art Painting)
        train_target_examples = []
        val_target_examples = []
        test_target_examples = []

        for category_idx, examples_list in target_examples.items():
            split_idx1 = round(target_category_ratios[category_idx] * val_split_length)
            split_idx2 = split_idx1 // 2
            for i, example in enumerate(examples_list):
                if i > split_idx1:
                    train_target_examples.append([example, category_idx])  # each pair is [path_to_img, class_label]
                elif split_idx2 <= i < split_idx1:
                    val_target_examples.append([example, category_idx])    # each pair is [path_to_img, class_label]
                else:
                    test_target_examples.append([example, category_idx])   # each pair is [path_to_img, class_label]

        # Transforms
        normalize = T.Normalize([0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ResNet18 - ImageNet Normalization

        train_transform = T.Compose([
            T.Resize(256),
            T.RandAugment(3, 15),
            T.CenterCrop(224),
            T.ToTensor(),
            normalize
        ])

        eval_transform = T.Compose([
            T.Resize(256),
            T.CenterCrop(224),
            T.ToTensor(),
            normalize
        ])

        # Dataloaders
        train_loader = DataLoader(PACSDatasetDisentangle(train_source_examples, train_target_examples, train_transform), batch_size=opt['batch_size'], num_workers=opt['num_workers'], shuffle=True)
        val_loader = DataLoader(PACSDatasetDisentangle(val_source_examples, val_target_examples, eval_transform), batch_size=opt['batch_size'], num_workers=opt['num_workers'], shuffle=False)
        test_loader = DataLoader(PACSDatasetDisentangle(test_source_examples, test_target_examples, eval_transform), batch_size=opt['batch_size'], num_workers=opt['num_workers'], shuffle=False)

        return train_loader, val_loader, test_loader

def build_splits_clip_disentangle(opt):

    source_domain = 'art_painting'
    target_domain = opt['target_domain']

    source_examples = read_lines(opt['data_path'], source_domain)
    target_examples = read_lines(opt['data_path'], target_domain)

    # SOURCE DOMAIN
    # Compute ratios of examples for each category
    source_category_ratios = {category_idx: len(examples_list) for category_idx, examples_list in
                              source_examples.items()}
    source_total_examples = sum(source_category_ratios.values())
    source_category_ratios = {category_idx: c / source_total_examples for category_idx, c in
                              source_category_ratios.items()}

    val_split_length = source_total_examples * 0.4  # 20% of the training split used for validation and 20% for test

    # Build splits - we train only on the source domain (Art Painting)
    train_source_examples = []
    val_source_examples = []
    test_source_examples = []

    for category_idx, examples_list in source_examples.items():
        split_idx1 = round(source_category_ratios[category_idx] * val_split_length)
        split_idx2 = split_idx1 // 2
        for i, example in enumerate(examples_list):
            if i > split_idx1:
                train_source_examples.append([example, category_idx])  # each pair is [path_to_img, class_label]
            elif split_idx2 <= i < split_idx1:
                val_source_examples.append([example, category_idx])  # each pair is [path_to_img, class_label]
            else:
                test_source_examples.append([example, category_idx])  # each pair is [path_to_img, class_label]

    # TARGET DOMAIN
    # Compute ratios of examples for each category
    target_category_ratios = {category_idx: len(examples_list) for category_idx, examples_list in
                              target_examples.items()}
    target_total_examples = sum(target_category_ratios.values())
    target_category_ratios = {category_idx: c / target_total_examples for category_idx, c in
                              target_category_ratios.items()}

    val_split_length = target_total_examples * 0.4  # 20% of the training split used for validation and 20% for test

    # Build splits - we train only on the source domain (Art Painting)
    train_target_examples = []
    val_target_examples = []
    test_target_examples = []

    for category_idx, examples_list in target_examples.items():
        split_idx1 = round(target_category_ratios[category_idx] * val_split_length)
        split_idx2 = split_idx1 // 2
        for i, example in enumerate(examples_list):
            if i > split_idx1:
                train_target_examples.append([example, category_idx])  # each pair is [path_to_img, class_label]
            elif split_idx2 <= i < split_idx1:
                val_target_examples.append([example, category_idx])  # each pair is [path_to_img, class_label]
            else:
                test_target_examples.append([example, category_idx])  # each pair is [path_to_img, class_label]

    # Transforms
    normalize = T.Normalize([0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ResNet18 - ImageNet Normalization

    train_transform = T.Compose([
        T.Resize(256),
        T.RandAugment(3, 15),
        T.CenterCrop(224),
        T.ToTensor(),
        normalize
    ])

    eval_transform = T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        normalize
    ])

    # Dataloaders
    train_loader = DataLoader(PACSDatasetClipDisentangle(train_source_examples, train_target_examples, train_transform, opt['data_path']),
                              batch_size=opt['batch_size'], num_workers=opt['num_workers'], shuffle=True,
                              )
    val_loader = DataLoader(PACSDatasetClipDisentangle(val_source_examples, val_target_examples, eval_transform, opt['data_path']),
                            batch_size=opt['batch_size'], num_workers=opt['num_workers'], shuffle=False,
                            )
    test_loader = DataLoader(PACSDatasetClipDisentangle(test_source_examples, test_target_examples, eval_transform, opt['data_path']),
                             batch_size=opt['batch_size'], num_workers=opt['num_workers'], shuffle=False,
                             )

    return train_loader, val_loader, test_loader

# DG
def build_splits_domain_generalization(opt):
    source_domains = SOURCE_DOMAINS[opt['target_domain']]
    separated_source_examples = []  # a list of dict, each dict is {key=category label : value=examples realated to the key} for each domain (three)
    domains_indeces = {}  # {0:[len of the lists of each category for the first domain], 1:[len of the lists of each category for the second domain], ...}
    print(f"tgt= {opt['target_domain']}, source_domains: {source_domains}")

    i = 0
    for source_domain in source_domains:
        tmp = read_lines(opt['data_path'], source_domain)
        tmp_list = []
        for k in tmp:
            tmp_list.append(len(tmp[k]))
        domains_indeces[i] = tmp_list
        separated_source_examples.append(tmp)
        i += 1
    # source_examples= {k: [d[k] for d in separated_source_examples] for k in separated_source_examples[0]} # dict comprehension
    source_examples = {}  # {key=category label : value= examples related to the key of all the domains}
    for k in separated_source_examples[0]:
        value = []
        for d in separated_source_examples:
            value += d[k]
        source_examples[k] = value

    target_domain = opt['target_domain']
    target_examples = read_lines(opt['data_path'], target_domain)

    # Compute ratios of examples for each category
    source_category_ratios = {category_idx: len(examples_list) for category_idx, examples_list in
                              source_examples.items()}
    source_total_examples = sum(source_category_ratios.values())
    source_category_ratios = {category_idx: c / source_total_examples for category_idx, c in
                              source_category_ratios.items()}

    # Build splits - we train only on the source domain (Art Painting)
    val_split_length = source_total_examples * 0.2  # 20% of the training split used for validation

    train_examples = []
    val_examples = []
    test_examples = []
    for category_idx, examples_list in source_examples.items():
        examples = []  # for each category, examples is a list of containing other lists of [example, category_label, domain_label]
        # according to the index, it assigns the different domain labels:
        for i, example in enumerate(examples_list):
            if i < domains_indeces[0][category_idx]:
                examples.append([example, category_idx, 0])
            elif domains_indeces[0][category_idx] <= i < domains_indeces[0][category_idx] + domains_indeces[1][
                category_idx]:
                examples.append([example, category_idx, 1])
            else:
                examples.append([example, category_idx, 2])
        random.Random(42).shuffle(
            examples)  # to divide in randomic way of domain_labels (because the first elements are all of domain =0, then =1 and then =2)
        split_idx = round(source_category_ratios[category_idx] * val_split_length)

        # divide the source examples in train and val
        for i, el in enumerate(examples):
            if i > split_idx:
                train_examples.append(el)
            else:
                val_examples.append(el)

    for category_idx, examples_list in target_examples.items():
        for example in examples_list:
            test_examples.append([example, category_idx, 3])  # each pair is [path_to_img, class_label, domain_label]

    # Transforms
    normalize = T.Normalize([0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ResNet18 - ImageNet Normalization

    train_transform = T.Compose([
        T.Resize(256),
        T.RandAugment(3, 15),
        T.CenterCrop(224),
        T.ToTensor(),
        normalize
    ])

    eval_transform = T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        normalize
    ])

    # Dataloaders
    train_loader = DataLoader(PACSDatasetDG(train_examples, train_transform), batch_size=opt['batch_size'],
                              num_workers=opt['num_workers'], shuffle=True)
    val_loader = DataLoader(PACSDatasetDG(val_examples, eval_transform), batch_size=opt['batch_size'],
                            num_workers=opt['num_workers'], shuffle=False)
    test_loader = DataLoader(PACSDatasetDG(test_examples, eval_transform), batch_size=opt['batch_size'],
                             num_workers=opt['num_workers'], shuffle=False)

    return train_loader, val_loader, test_loader

# DG
def build_splits_domain_generalization_clip(opt):
    source_domains = SOURCE_DOMAINS[opt['target_domain']]
    separated_source_examples = []  # a list of dict, each dict is {key=category label : value=examples realated to the key} for each domain (three)
    domains_indeces = {}  # {0:[len of the lists of each category for the first domain], 1:[len of the lists of each category for the second domain], ...}
    print(f"tgt= {opt['target_domain']}, source_domains: {source_domains}")

    i = 0
    for source_domain in source_domains:
        tmp = read_lines(opt['data_path'], source_domain)
        tmp_list = []
        for k in tmp:
            tmp_list.append(len(tmp[k]))
        domains_indeces[i] = tmp_list
        separated_source_examples.append(tmp)
        i += 1
    # source_examples= {k: [d[k] for d in separated_source_examples] for k in separated_source_examples[0]} # dict comprehension
    source_examples = {}  # {key=category label : value= examples related to the key of all the domains}
    for k in separated_source_examples[0]:
        value = []
        for d in separated_source_examples:
            value += d[k]
        source_examples[k] = value

    target_domain = opt['target_domain']
    target_examples = read_lines(opt['data_path'], target_domain)

    # Compute ratios of examples for each category
    source_category_ratios = {category_idx: len(examples_list) for category_idx, examples_list in
                              source_examples.items()}
    source_total_examples = sum(source_category_ratios.values())
    source_category_ratios = {category_idx: c / source_total_examples for category_idx, c in
                              source_category_ratios.items()}

    # Build splits - we train only on the source domain (Art Painting)
    val_split_length = source_total_examples * 0.2  # 20% of the training split used for validation

    train_examples = []
    val_examples = []
    test_examples = []
    for category_idx, examples_list in source_examples.items():
        examples = []  # for each category, examples is a list of containing other lists of [example, category_label, domain_label]
        # according to the index, it assigns the different domain labels:
        for i, example in enumerate(examples_list):
            if i < domains_indeces[0][category_idx]:
                examples.append([example, category_idx, 0])
            elif domains_indeces[0][category_idx] <= i < domains_indeces[0][category_idx] + domains_indeces[1][
                category_idx]:
                examples.append([example, category_idx, 1])
            else:
                examples.append([example, category_idx, 2])
        random.Random(42).shuffle(
            examples)  # to divide in randomic way of domain_labels (because the first elements are all of domain =0, then =1 and then =2)
        split_idx = round(source_category_ratios[category_idx] * val_split_length)

        # divide the source examples in train and val
        for i, el in enumerate(examples):
            if i > split_idx:
                train_examples.append(el)
            else:
                val_examples.append(el)

    for category_idx, examples_list in target_examples.items():
        for example in examples_list:
            test_examples.append([example, category_idx, 3])  # each pair is [path_to_img, class_label, domain_label]

    # Transforms
    normalize = T.Normalize([0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ResNet18 - ImageNet Normalization

    train_transform = T.Compose([
        T.Resize(256),
        T.RandAugment(3, 15),
        T.CenterCrop(224),
        T.ToTensor(),
        normalize
    ])

    eval_transform = T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        normalize
    ])

    # Dataloaders
    train_loader = DataLoader(PACSDatasetDG_Clip(train_examples, train_transform, opt['data_path']), batch_size=opt['batch_size'],
                              num_workers=opt['num_workers'], shuffle=True)
    val_loader = DataLoader(PACSDatasetDG_Clip(val_examples, eval_transform, opt['data_path']), batch_size=opt['batch_size'],
                            num_workers=opt['num_workers'], shuffle=False)
    test_loader = DataLoader(PACSDatasetDG_Clip(test_examples, eval_transform, opt['data_path']), batch_size=opt['batch_size'],
                             num_workers=opt['num_workers'], shuffle=False)

    return train_loader, val_loader, test_loader



def get_target_data(opt):
    target_domain = opt['target_domain']
    target_examples = read_lines(opt['data_path'], target_domain)

    test_examples = []
    for category_idx, examples_list in target_examples.items():
        for example in examples_list:
            test_examples.append([example, category_idx])  # each pair is [path_to_img, class_label]

    # Transforms
    normalize = T.Normalize([0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ResNet18 - ImageNet Normalization

    eval_transform = T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        normalize
    ])

    # Dataloaders
    test_loader = DataLoader(PACSDatasetBaseline(test_examples, eval_transform), batch_size=opt['batch_size'],
                             num_workers=opt['num_workers'], shuffle=False)

    return test_loader
