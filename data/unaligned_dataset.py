import os.path
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from PIL import Image
import random
import util.util as util
import torchvision.transforms as transforms


def move(imageDir,img_name,x,y): #平移，平移尺度为off
    img = Image.open(os.path.join(imageDir, img_name))
    offset = ImageChops.offset(img,x,y)
    return offset

class UnalignedDataset(BaseDataset):
    """
    This dataset class can load unaligned/unpaired datasets.

    It requires two directories to host training images from domain A '/path/to/data/trainA'
    and from domain B '/path/to/data/trainB' respectively.
    You can train the model with the dataset flag '--dataroot /path/to/data'.
    Similarly, you need to prepare two directories:
    '/path/to/data/testA' and '/path/to/data/testB' during test time.
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        self.opt = opt
        BaseDataset.__init__(self, opt)
        # self.dir_A = os.path.join(opt.dataroot, opt.phase + 'A')  # create a path '/path/to/data/trainA'
        # self.dir_B = os.path.join(opt.dataroot, opt.phase + 'B')  # create a path '/path/to/data/trainB'
        
        if opt.phase == "train":
            self.dir_A = os.path.join(opt.dataroot, opt.phase, 'haze')  # create a path '/path/to/data/train/haze'
            self.dir_B = os.path.join(opt.dataroot, opt.phase, 'ref')  # create a path '/path/to/data/train/clean'
            self.dir_C = os.path.join(opt.dataroot, opt.phase, 'dehaze_dcp')  #  create a path '/path/to/data/train/dehaze_ffa_resize'
            self.dir_D = os.path.join(opt.dataroot, opt.phase, 'dehaze_ffa_resize')
            self.dir_E = os.path.join(opt.dataroot, opt.phase, 'dehaze_c2pnet_resize')

        # if opt.phase == "test" and not os.path.exists(self.dir_A) \
           # and os.path.exists(os.path.join(opt.dataroot, "valA")):
            # self.dir_A = os.path.join(opt.dataroot, "valA")
            # self.dir_B = os.path.join(opt.dataroot, "valB")
            
        if opt.phase == "test":
            # self.dir_A = os.path.join(opt.dataroot, 'haze')  # create a path '/path/to/data/train/haze'
            # self.dir_B = os.path.join(opt.dataroot, 'gt500')  # create a path '/path/to/data/train/clean'
            # self.dir_B = os.path.join(opt.dataroot, 'ref') 
            
            self.dir_A = os.path.join(opt.dataroot, opt.phase, 'haze')  # create a path '/path/to/data/train/haze'
            self.dir_B = os.path.join(opt.dataroot, opt.phase, 'ref')  # create a path '/path/to/data/train/clean'
            self.dir_C = os.path.join(opt.dataroot, opt.phase, 'dehaze_dcp')
            self.dir_D = os.path.join(opt.dataroot, opt.phase, 'dehaze_ffa_resize')
            self.dir_E = os.path.join(opt.dataroot, opt.phase, 'dehaze_c2pnet_resize')
            
            # self.dir_A = os.path.join(opt.dataroot, 'hazy')
            # self.dir_B = os.path.join(opt.dataroot, 'hazy')

        self.A_paths = sorted(make_dataset(self.dir_A, opt.max_dataset_size))   # load images from '/path/to/data/train/haze'
        self.B_paths = sorted(make_dataset(self.dir_B, opt.max_dataset_size))    # load images from '/path/to/data/train/clean'
        self.C_paths = sorted(make_dataset(self.dir_C, opt.max_dataset_size)) 
        self.D_paths = sorted(make_dataset(self.dir_D, opt.max_dataset_size)) 
        self.E_paths = sorted(make_dataset(self.dir_E, opt.max_dataset_size)) 
        self.A_size = len(self.A_paths)  # get the size of dataset A
        self.B_size = len(self.B_paths)  # get the size of dataset B
        self.C_size = len(self.C_paths)
        self.D_size = len(self.D_paths)
        self.E_size = len(self.E_paths)

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index (int)      -- a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor)       -- an image in the input domain
            B (tensor)       -- its corresponding image in the target domain
            A_paths (str)    -- image paths
            B_paths (str)    -- image paths
        """
        '''
        # ref is got by random
        if self.opt.phase == "train":
            A_path = self.A_paths[index % self.A_size]  # make sure index is within then range
            if self.opt.serial_batches:   # make sure index is within then range
                index_B = index % self.B_size
            else:   # randomize the index for domain B to avoid fixed pairs.
                index_B = random.randint(0, self.B_size - 1)
            B_path = self.B_paths[index_B]
            
            A_img = Image.open(A_path).convert('RGB')
            B_img = Image.open(B_path).convert('RGB')

            # Apply image transformation
            # For CUT/FastCUT mode, if in finetuning phase (learning rate is decaying),
            # do not perform resize-crop data augmentation of CycleGAN.
            is_finetuning = self.opt.isTrain and self.current_epoch > self.opt.n_epochs
            modified_opt = util.copyconf(self.opt, load_size=self.opt.crop_size if is_finetuning else self.opt.load_size)
            transform = get_transform(modified_opt)
            # transform = get_transform(self.opt)
            A = transform(A_img)
            B = transform(B_img)
            
         # ref img is got by the crop and resize
         if self.opt.phase == "train":
            A_path = self.A_paths[index % self.A_size]  # make sure index is within then range
            
            B_path = self.B_paths[index % self.B_size]
            A_img = Image.open(A_path).convert('RGB')
            B_img = Image.open(B_path).convert('RGB')
            
            # use the gt to generate the reference image
            width_B_img, height_B_img = B_img.size
            scale = 0.8
            x, y = random.randrange(0, width_B_img - int(width_B_img*scale) + 1), random.randrange(0, height_B_img - int(height_B_img*scale) + 1)
            B_img = B_img.crop((x, y, x + int(width_B_img*scale), y + int(height_B_img*scale)))
            B_img = B_img.resize((width_B_img, height_B_img), Image.ANTIALIAS)
            
            if B_img.size[0] != A_img.size[0] or B_img.size[1] != A_img.size[1] :
                print("the size of input image is different from the ref image")

            transform = get_transform(self.opt)
            A = transform(A_img)
            B = transform(B_img)
        '''
        # ref img is from the MINE
        if self.opt.phase == "train":
            A_path = self.A_paths[index % self.A_size]  # make sure index is within then range
            # if self.opt.serial_batches:   # make sure index is within then range
                # index_B = index % self.B_size
            # else:   # randomize the index for domain B to avoid fixed pairs.
                # index_B = random.randint(0, self.B_size - 1)
            # B_path = self.B_paths[index_B]
            
            B_path = self.B_paths[index % self.B_size]
            C_path = self.C_paths[index % self.C_size]
            D_path = self.D_paths[index % self.D_size]
            E_path = self.E_paths[index % self.E_size]
            A_img = Image.open(A_path).convert('RGB')
            B_img = Image.open(B_path).convert('RGB')
            C_img = Image.open(C_path).convert('RGB')
            D_img = Image.open(D_path).convert('RGB')
            E_img = Image.open(E_path).convert('RGB')
            
            
            # for ots dataset:
            # width, height = B_img.size
        
            # if width < self.size or height < self.size:
                # if width < height:
                    # A_img = A_img.resize((self.size+4, int((self.size+4) * (height / width))), Image.ANTIALIAS)
                    # B_img = B_img.resize((self.size+4, int((self.size+4) * (height / width))), Image.ANTIALIAS)
                # else:
                    # A_img = A_img.resize((int((self.size+4) * (width / height)), self.size+4), Image.ANTIALIAS)
                    # B_img = B_img.resize((int((self.size+4) * (width / height)), self.size+4), Image.ANTIALIAS)
                # width, height = B_img.size
            # x, y = random.randrange(0, width - self.size + 1), random.randrange(0, height - self.size + 1)
            # A_img = A_img.crop((x, y, x + self.size, y + self.size))
            # B_img = B_img.crop((x, y, x + self.size, y + self.size))
            
            
            
            # for the ots2：
            A_img = A_img.resize((self.opt.load_size, self.opt.load_size), Image.ANTIALIAS)
            B_img = B_img.resize((self.opt.load_size, self.opt.load_size), Image.ANTIALIAS)
            C_img = C_img.resize((self.opt.load_size, self.opt.load_size), Image.ANTIALIAS)
            D_img = D_img.resize((self.opt.load_size, self.opt.load_size), Image.ANTIALIAS)
            E_img = E_img.resize((self.opt.load_size, self.opt.load_size), Image.ANTIALIAS)
            
            
            #for the its dataset:
            width, height = B_img.size
            x, y = random.randrange(0, width - int(self.opt.crop_size) + 1), random.randrange(0, height - int(self.opt.crop_size) + 1)
            A_img = A_img.crop((x, y, x + int(self.opt.crop_size), y + int(self.opt.crop_size)))
            B_img = B_img.crop((x, y, x + int(self.opt.crop_size), y + int(self.opt.crop_size)))
            C_img = C_img.crop((x, y, x + int(self.opt.crop_size), y + int(self.opt.crop_size)))
            D_img = D_img.crop((x, y, x + int(self.opt.crop_size), y + int(self.opt.crop_size)))
            E_img = E_img.crop((x, y, x + int(self.opt.crop_size), y + int(self.opt.crop_size)))
            
            
            if B_img.size[0] != A_img.size[0] or B_img.size[1] != A_img.size[1] :
                print("the size of input image is different from the ref image")

            # Apply image transformation
            # For CUT/FastCUT mode, if in finetuning phase (learning rate is decaying),
            # do not perform resize-crop data augmentation of CycleGAN.
            # is_finetuning = self.opt.isTrain and self.current_epoch > self.opt.n_epochs
            # modified_opt = util.copyconf(self.opt, load_size=self.opt.crop_size if is_finetuning else self.opt.load_size)
            # transform = get_transform(modified_opt)
            # transform = get_transform(self.opt)
            # transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
            transform = transforms.Compose([transforms.ToTensor()])
            A = transform(A_img)
            B = transform(B_img)
            C = transform(C_img)
            D = transform(D_img)
            E = transform(E_img)
       
        if self.opt.phase == "test":
       
            A_path = self.A_paths[index]
            
            # random reference 
            index_B = random.randint(0, self.B_size - 1)
            B_path = self.B_paths[index_B]
            
            # B_path = self.B_paths[index]
            C_path = self.C_paths[index]
            D_path = self.D_paths[index]
            E_path = self.E_paths[index]
            A_img = Image.open(A_path).convert('RGB')
            B_img = Image.open(B_path).convert('RGB')
            C_img = Image.open(C_path).convert('RGB')
            D_img = Image.open(D_path).convert('RGB')
            E_img = Image.open(E_path).convert('RGB')
            
            # for the o-hazy and i-hazy：
            # width, height = B_img.size            
            # A_img = A_img.resize((int(width/4), int(height/4)), Image.ANTIALIAS)
            # B_img = B_img.resize((int(width/4), int(height/4)), Image.ANTIALIAS)
            
            # for the hsts_real：
            # width, height = B_img.size
            # if width > 1500 and height > 1500:       
                # A_img = A_img.resize((int(width/2), int(height/2)), Image.ANTIALIAS)
                # B_img = B_img.resize((int(width/2), int(height/2)), Image.ANTIALIAS)
                # C_img = C_img.resize((int(width/2), int(height/2)), Image.ANTIALIAS)
                # D_img = D_img.resize((int(width/2), int(height/2)), Image.ANTIALIAS)
                # E_img = E_img.resize((int(width/2), int(height/2)), Image.ANTIALIAS)
            
            
            
            width, height = A_img.size            
            A_img = A_img.resize((int(width/2), int(height/2)), Image.ANTIALIAS)
            B_img = B_img.resize((int(width/2), int(height/2)), Image.ANTIALIAS)
            
            # for the 1st type
            # input_nc = self.opt.output_nc if self.opt.direction == 'BtoA' else self.opt.input_nc
            # transform_input = get_transform(self.opt, grayscale=(input_nc == 1))
            # A = transform_input(A_img)
            # B = transform_input(B_img)
            
            
           # for the 2nd type
            # width, height = B_img.size 
            # A_img = A_img.resize((int(round(width/4)*4), int(round(height/4)*4)), Image.ANTIALIAS)
            # B_img = B_img.resize((int(round(width/4)*4), int(round(height/4)*4)), Image.ANTIALIAS)
            
            # test_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
            test_transform = transforms.Compose([transforms.ToTensor()])
            A = test_transform(A_img)
            B = test_transform(B_img)
            C = test_transform(C_img)
            D = test_transform(D_img)
            E = test_transform(E_img)
            print(A.shape)
            
            
        return {'A': A, 'B': B, 'C': C, 'D': D, 'E': E, 'A_paths': A_path, 'B_paths': B_path}
        # return {'A': A, 'B': B, 'C': C, 'D': D, 'A_paths': A_path, 'B_paths': B_path}

    def __len__(self):
        """Return the total number of images in the dataset.

        As we have two datasets with potentially different number of images,
        we take a maximum of
        """
        return max(self.A_size, self.B_size)
