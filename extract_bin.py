
class extract_bin:
    def __init__(self, file_path = "amazon_cells_labelled.txt"):

        # Initialize lists for inputs and labels
        inputs = []
        labels = []

        line_ctr = 0
        lines_list = []
        # Read the file and process each line
        with open(file_path, "r") as file:
            for line in file:
                line_ctr = line_ctr + 1
                line = line.strip()  # Remove any leading/trailing whitespace
                if not line:
                    continue  # Skip empty lines            
                
                sentence, sentiment = line[:-1], int(line[-1])
                
                # Append the sentence and corresponding label
                inputs.append(sentence)
                labels.append(sentiment)

        self.inputs = inputs
        self.labels = labels
        self.lines_list = lines_list

class split:
    def __init__(self, inputs, labels, train_size):
        if( train_size > len(inputs)-1 ):
            train_size = int(len(inputs)*0.9)
        
        self.train_set_inputs = inputs[:train_size]
        self.train_set_labels = labels[:train_size]
        self.test_set_inputs = inputs[train_size:]
        self.test_set_labels = labels[train_size:]
