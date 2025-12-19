from filesystem.file_io import FileIO 
from preprocessor import Preprocessor
from splitter import Splitter
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils.data import Subset
from dataset import BreastCancerDataset
from pathlib import Path
from filesystem.file_comparison_tool import FileComparisonTool
from dataset import BreastCancerDataset
# from tester import Tester
# from trainer import Trainer
from tester_with_graphs import TesterWithGraphs
from trainer_with_graphs import TrainerWithGraphs
from sequence_generator import SequenceGenerator
from sequence_generator_labeled import SequenceGeneratorLabeled
from sequence_generator_3rows import SequenceGenerator3Rows


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

    """
    # train_dataset = BreastCancerDataset(root_dir = "data_split", split = "train")
    #train_dataset = BreastCancerDataset(root_dir = "data_split_tiny_fake", split = "train")
    #train_dataset = BreastCancerDataset(root_dir = "data_split_tiny", split = "train")
    train_dataset = BreastCancerDataset(root_dir = "data_split", split = "train")
    validation_dataset = BreastCancerDataset(root_dir = "data_split", split = "validation")
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    validation_loader = DataLoader(validation_dataset, batch_size=32, shuffle=True)


    FileComparisonTool.compare_all_files_local(
        "data_split/train",
        "data_split/validation",
    )

    FileComparisonTool.compare_all_images_local(
        "data_split/train",
        "data_split/validation",
        extension="png",
    )
    
    TrainerWithGraphs.train(train_loader, validation_loader, epochs=10)
    """

    
    test_dataset = BreastCancerDataset(root_dir = "data_split", split = "test")
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)
    TesterWithGraphs.test(test_loader, model_subdir="training_run", model_file_name="latest_0050")

    SequenceGenerator.generate(
    loader=test_loader,
    model_subdir="training_run",
    model_file_name="latest_0050",
    out_name="sequence_latest_0050"
    )
    
    SequenceGeneratorLabeled.generate(
    loader=test_loader,
    model_subdir="training_run",
    model_file_name="latest_0050",
    out_name="sequence_labeled_latest_0050"
    )

    SequenceGenerator3Rows.generate(
        loader=test_loader,
        model_subdir="training_run",
        model_file_name="latest_0050",
        out_name="sequence_3rows_latest_0050",
        label="BreastCancerClassifier signal flow",
    )
    
if __name__ == "__main__":
    main()