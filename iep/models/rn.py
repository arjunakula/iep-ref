import torch
import torch.nn as nn
from torch.autograd import Variable

from iep.models.layers import LstmEncoder


def feature_pairs(feats):
  """
  Input:
  - feats: N x M x D
  
  Output:
  - pairs: N x M^2 x D
  """
  is_var = isinstance(feats, Variable)
  dtype = feats.data.type() if is_var else feats.type()
  N, M, D = feats.size()
  idx = torch.arange(0, M)
  idx1 = idx.repeat(M, 1).t().contiguous().view(-1).type(dtype).long()
  if is_var:
    idx1 = Variable(idx1)
  feats1 = feats.index_select(1, idx1)
  feats2 = feats.repeat(1, M, 1)
  pairs = torch.cat([feats1, feats2], 2)
  return pairs


class RelationNetworkModel(nn.Module):
  def __init__(self, vocab,
      rnn_wordvec_dim=32, rnn_dim=128, rnn_num_layers=1, rnn_dropout=0,
      cnn_dim=24, cnn_num_layers=4, cnn_stride=2,
      g_dim=256, g_num_layers=4,
      f_dim=256, f_num_layers=3, f_dropout=0.5):
    super(RelationNetworkModel, self).__init__()
   
    rnn_kwargs = {
      'token_to_idx': vocab['question_token_to_idx'],
      'wordvec_dim': rnn_wordvec_dim,
      'rnn_dim': rnn_dim,
      'rnn_num_layers': rnn_num_layers,
      'rnn_dropout': rnn_dropout,
    }
    self.rnn = LstmEncoder(**rnn_kwargs)
    self.cnn = self._build_cnn(cnn_num_layers, cnn_dim, cnn_stride)

    g_input_dim = 2 * (cnn_dim + 2) + rnn_dim
    g_layers = [nn.Linear(g_input_dim, g_dim), nn.ReLU(inplace=True)]
    for _ in range(g_num_layers - 1):
      g_layers.append(nn.Linear(g_dim, g_dim))
      g_layers.append(nn.ReLU(inplace=True))
    self.g = nn.Sequential(*g_layers)

    num_answers = len(vocab['answer_token_to_idx'])
    f_layers = [nn.Linear(g_dim, f_dim), nn.ReLU(inplace=True)]
    for _ in range(f_num_layers - 2):
      f_layers.append(nn.Linear(f_dim, f_dim))
      f_layers.append(nn.ReLU(inplace=True))
      if f_dropout > 0:
        f_layers.append(nn.Dropout(f_dropout))
    f_layers.append(nn.Linear(f_dim, num_answers))
    self.f = nn.Sequential(*f_layers)

  def _build_cnn(self, cnn_num_layers, cnn_dim, cnn_stride):
    layers = []
    prev_dim = 3
    for i in range(cnn_num_layers):
      layers.append(nn.Conv2d(prev_dim, cnn_dim, kernel_size=3, padding=1,
                              stride=cnn_stride))
      layers.append(nn.BatchNorm2d(cnn_dim))
      layers.append(nn.ReLU(inplace=True))
      prev_dim = cnn_dim
    return nn.Sequential(*layers)

  def forward(self, questions, images):
    q_feats = self.rnn(questions)
    img_feats = self.cnn(images)
    N, C, H, W = img_feats.size()
    M = H * W

    x_coords = torch.linspace(0, 1, W).view(1, 1, 1, W).expand(N, 1, H, W)
    y_coords = torch.linspace(0, 1, H).view(1, 1, H, 1).expand(N, 1, H, W)
    x_coords = Variable(x_coords.type_as(images.data))
    y_coords = Variable(y_coords.type_as(images.data))

    objects = torch.cat([img_feats, x_coords, y_coords], 1)
    objects = objects.view(N, C + 2, H * W).transpose(1, 2)
    object_pairs = feature_pairs(objects)

    q_feats_rep = q_feats[:, None].repeat(1, M * M, 1)
    g_input = torch.cat([object_pairs, q_feats_rep], 2)

    g_output = self.g(g_input.view(N * M * M, -1)).view(N, M * M, -1)
    f_input = g_output.sum(1)[:, 0]
    scores = self.f(f_input)

    return scores


if __name__ == '__main__':
  from iep.utils import load_vocab
  vocab = load_vocab('data/vocab.json')
  model = RelationNetworkModel(vocab).cuda()
  loss_fn = nn.CrossEntropyLoss().cuda()

  N, H, W, L = 64, 128, 128, 10
  images = torch.randn(N, 3, H, W)
  questions = torch.LongTensor(N, L).random_(5)
  answers = torch.LongTensor(N).random_(10)
  
  images = Variable(images.cuda())
  questions = Variable(questions.cuda())
  answers = Variable(answers.cuda())

  for t in range(10):
    scores = model(questions, images)
    loss = loss_fn(scores, answers)
    loss.backward()

