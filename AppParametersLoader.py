import argparse

class AppParametersLoader:
    def __init__(self):
        #ArgumentParser init
        self.parser = argparse.ArgumentParser()
        #---------------------------training parameters ---------------------------
        self.parser.add_argument('--save_dir',
                                 action = 'store',
                                 default = 'ImageClassifier/checkpoints',
                                 type = str,
                                 help ='directory for the checkpoint')
        
        self.parser.add_argument('--data_dir',
                                 action = 'store',
                                 default = 'ImageClassifier/flowers',
                                 type = str,
                                 help ='directory fro data')
        
        self.parser.add_argument('--arch',
                                 action = 'store',
                                 default = 'vgg16',
                                 help = 'used architecture: pre-trained model')
        

        self.parser.add_argument('--learning_rate',
                                 action='store',
                                 default=0.001,
                                 type=float,
                                 help='learning rate used in weights update')
        
        self.parser.add_argument('--hidden_units',
                                 action ='store',
                                 default = 5000,
                                 type = int,
                                 help = 'number of hidden layers for the classifier')


        self.parser.add_argument('--epochs',
                                 action ='store',
                                 type = int,
                                 default = 5,
                                 help = 'how many epochs for training')

        self.parser.add_argument('--gpu',
                                 action = 'store_true',
                                 dest = 'gpu',
                                 default = True,
                                 help = 'Use GPU for training')

        #---------------------------prediction parameters ---------------------------
        self.parser.add_argument('--image',
                                 action = 'store',
                                 dest = 'image',
                                 default = 'ImageClassifier/flowers/test/16/image_06657.jpg',
                                 help = 'path to image')
        
        self.parser.add_argument('--top_k',
                                 action = 'store',
                                 type = int,
                                 default = 5,
                                 help = 'how many k probs and classes')
        
        self.parser.add_argument('--category_names',
                                 action = 'store',
                                 type = str,
                                 default = 'ImageClassifier/cat_to_name.json',
                                 help = 'file json with category_names')
        #parsing arguments
        self.args = self.parser.parse_args()
    
    #public getter methods for Application Parameters.
    def save_dir(self):
         return self.args.save_dir

    def data_dir(self):
         return self.args.data_dir
    
    def arch(self):
        return self.args.arch
    
    def learning_rate(self):
        return self.args.learning_rate
    
    def hidden_units(self):
         return self.args.hidden_units
    
    def epochs(self):
        return self.args.epochs
    
    def gpu(self):
        return self.args.gpu
    
    def image_path(self):
        return self.args.image
    
    def top_k(self):
        return self.args.top_k
    
    def category_names_path(self):
         return self.args.category_names
    
    #parameters print statement
    def print_all(self):
        print(f"Parameters: "
              f"save_dir: {self.save_dir()} ,"
              f"data_dir: {self.data_dir()} ,"
              f"arch: {self.arch()} ,"
              f"learning_rate: {self.learning_rate()} ,"
              f"hidden_units: {self.hidden_units()} ,"
              f"epochs: {self.epochs()} ,"
              f"gpu: {self.gpu()} ,"
              f"image_path: {self.image_path()} ..."
              f"top_k: {self.top_k()} ,"
              f"save_dir: {self.category_names_path()} ...")

    