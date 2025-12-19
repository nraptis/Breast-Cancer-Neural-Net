from filesystem.file_io import FileIO 
from preprocessor import Preprocessor
from splitter import Splitter
from trainer import Trainer
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils.data import Subset
from dataset import BreastCancerDataset

def main():

    """
    data_original_benign = FileIO.get_all_files_local_recursive("data_original/benign")
    data_original_malignant = FileIO.get_all_files_local_recursive("data_original/malignant")

    for f in data_original_benign[:10]:
        print(f)
    
    print("---")

    for f in data_original_malignant[:10]:
        print(f)
    
    print("---")

    Preprocessor.go(data_original_benign, data_original_malignant)
    

    
    data_processed_benign = FileIO.get_all_files_local_recursive("data_processed/benign")
    data_processed_malignant = FileIO.get_all_files_local_recursive("data_processed/malignant")

    Splitter.go(data_processed_benign, data_processed_malignant)
    """

    # train_dataset = BreastCancerDataset(root_dir = "data_split", split = "train")
    #train_dataset = BreastCancerDataset(root_dir = "data_split_tiny_fake", split = "train")
    #train_dataset = BreastCancerDataset(root_dir = "data_split_tiny", split = "train")
    train_dataset = BreastCancerDataset(root_dir = "data_split", split = "train")
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    Trainer.train(train_loader, epochs=1024)
    
if __name__ == "__main__":
    main()