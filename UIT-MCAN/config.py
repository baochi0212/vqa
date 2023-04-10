# paths
qa_path = './ViVQA'  # directory containing the question and annotation jsons
train_path = './ViVQA/train'  # directory of training images
# val_path = '../ViVQA/val'  # directory of validation images
val_path = ""
test_path = './ViVQA/test'  # directory of test images
vocabulary_path = 'vocab.json'  # path where the used vocabularies for question and answers are saved to
preprocessed_path = './resnet-14x14.h5'  # path where preprocessed features are saved to and loaded from
json_train_path = "./ViVQA/vivqa_train_2017.json"
json_test_path = "./ViVQA/vivqa_test_2017.json"
json_overfit_path = "./ViVQA/vivqa_overfitset_2017.json"
image_size = (448, 448)
train_image_dir = "./ViVQA/images/train"
test_image_dir = "./ViVQA/images/test"
image_patch_size = 32
task = 'OpenEnded'
dataset = 'ViVQA'

# training config
epochs = 30
batch_size = 64
initial_lr = 5e-5  # default Adam lr
lr_halflife = 50000  # in iterations
data_workers = 0
model_checkpoint = "./saved_models"
best_model_checkpoint = "./saved_models"
tmp_model_checkpoint = "./saved_models"
start_from = "./saved_models/model_best.pth"
backbone = "resnet152"

## self-attention based method configurations
d_model = 512
embedding_dim = 300
dff = 1024
nheads = 8
nlayers = 4
dropout = 0.5
word_embedding = "fasttext.vi.300d"

#log infer
log_file = './log_inference'