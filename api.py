import os
import numpy as np
import pandas as pd

class MagRetraceData:
    def __init__(self, folder_root):
        self.folder_root = folder_root
        self.experiment_start_time = self._read_experiment_start_time()
        self.time_seq, self.mag_x, self.mag_y, self.mag_z, self.mag_abs = self._read_data()
    
    # Functionalities that reads-in data from the source.
    def _read_experiment_start_time(self):
        csv_path = os.path.join(self.folder_root, "meta", "time.csv")
        df = pd.read_csv(csv_path, index_col=0)
        return df.loc['START', 'system time text']
    
    def _read_data(self):
        csv_path = os.path.join(self.folder_root, "Raw Data.csv")
        df = pd.read_csv(csv_path)
        time_seq = df['Time (s)'].values
        mag_x = df['Magnetic Field x (µT)'].values
        mag_y = df['Magnetic Field y (µT)'].values
        mag_z = df['Magnetic Field z (µT)'].values
        mag_abs = df['Absolute field (µT)'].values
        return time_seq, mag_x, mag_y, mag_z, mag_abs


class MagRetraceDataAPI:
    def __init__(self, data_root):
        self.data_root = data_root
        self.template_root = os.path.join(self.data_root, "templates")
        self.traversal_root = os.path.join(self.data_root, "traversals")
        self.template_list = self._get_template_list()
        self.traversal_list = self._get_traversal_list()
    
    def _get_template_list(self):
        template_list = os.listdir(self.template_root)
        template_list = sorted(template_list)
        nondata_files = [".DS_Store"]
        for nondata_file in nondata_files:
            if nondata_file in template_list:
                template_list.remove(nondata_file)
        return template_list
    
    def _get_traversal_list(self):
        traversal_list = os.listdir(self.traversal_root)
        traversal_list = sorted(traversal_list)
        nondata_files = [".DS_Store"]
        for nondata_file in nondata_files:
            if nondata_file in traversal_list:
                traversal_list.remove(nondata_file)
        return traversal_list
        
    # Listing modules with "printing" functionalities.
    def list_template_titles(self):
        print("Available template titles:")
        for idx in range(len(self.template_list)):
            title = self.template_list[idx]
            print(f"  {idx} - {title}")
    
    def list_traversal_titles(self):
        print("Available traversal titles:")
        for idx in range(len(self.traversal_list)):
            title = self.traversal_list[idx]
            print(f"  {idx} - {title}")
    
    def get_template_data(self, identifier) -> MagRetraceData:
        if isinstance(identifier, int):
            data_title = self.template_list[identifier]
        elif isinstance(identifier, str):
            data_title = identifier
        else:
            raise TypeError("Identifier must be either an integer or a string.")
        data_path = os.path.join(self.data_root, "templates", data_title)
        return MagRetraceData(data_path)
    
    def get_traversal_data(self, identifier) -> MagRetraceData:
        if isinstance(identifier, int):
            data_title = self.traversal_list[identifier]
        elif isinstance(identifier, str):
            data_title = identifier
        else:
            raise TypeError("Identifier must be either an integer or a string.")
        data_path = os.path.join(self.data_root, "traversals", data_title)
        return MagRetraceData(data_path)



