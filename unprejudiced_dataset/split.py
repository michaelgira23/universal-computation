import numpy as np
from sklearn.model_selection import train_test_split

# Read lines from dataset
file = open('combined.txt', 'r')
nonempty_lines = [line.strip('\n') for line in file if line != '\n']
line_count = len(nonempty_lines)
file.close()

# Pair corresponding sentences together
corresponding_sentences = np.array(nonempty_lines).reshape(int(line_count / 2), 2)
labels = np.zeros(int(line_count / 2))

# Split pairs into test and training
X_train, X_test, y_train, y_test = train_test_split(corresponding_sentences, labels, train_size=0.8, random_state=42)

# Write to separate files
file = open('train.txt', 'w')
file.writelines('\n'.join(X_train.flatten().tolist()))
file.close()

file = open('test.txt', 'w')
file.writelines('\n'.join(X_test.flatten().tolist()))
file.close()
