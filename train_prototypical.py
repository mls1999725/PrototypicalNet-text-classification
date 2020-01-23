import json

import torch
from data_helper import PrototypicalData
from metrics import get_multi_metrics, mean
from Prototypical import PrototypicalNet


class PrototypicalTrainer(object):
    def __init__(self, config_path):
        with open(config_path, "r") as fr:
            self.config = json.load(fr)

        self.train_data_obj = self.load_data()
        self.eval_data_obj = self.load_data(is_training=False)
        self.train_tasks = self.train_data_obj.gen_data(self.config["train_data"])
        self.eval_tasks = self.eval_data_obj.gen_data(self.config["eval_data"])

        print("vocab size: ", self.train_data_obj.vocab_size)

        self.model = self.create_model()

    def load_data(self, is_training=True):
        """
        init data object
        :return:
        """
        data_obj = PrototypicalData(output_path=self.config["output_path"],
                                    sequence_length=self.config["sequence_length"],
                                    num_classes=self.config["num_classes"],
                                    num_support=self.config["num_support"],
                                    num_queries=self.config["num_queries"],
                                    num_tasks=self.config["num_tasks"],
                                    num_eval_tasks=self.config["num_eval_tasks"],
                                    embedding_size=self.config["embedding_size"],
                                    stop_word_path=self.config["stop_word_path"],
                                    word_vector_path=self.config["word_vector_path"],
                                    is_training=is_training)
        return data_obj

    def create_model(self):
        """
        init model object
        :return:
        """
        model = PrototypicalNet(config=self.config)
        return model

    def one_hot(self, labels):
        labels_empty = torch.zeros(self.config["num_classes"] * self.config["num_queries"],
                                   self.config["num_classes"])
        for i in range(labels.size(0)):
            labels_empty[i][labels[i]] = 1

        return labels_empty

    def train(self):
        """
        train model
        :return:
        """
        current_step = 0
        loss_function = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config["learning_rate"])
        for epoch in range(self.config["epochs"]):
            print("----- Epoch {}/{} -----".format(epoch + 1, self.config["epochs"]))
            for batch in self.train_data_obj.next_batch(self.train_tasks):
                predictions = self.model(batch)
                # print(predictions)
                optimizer.zero_grad()
                labels = torch.LongTensor(batch["labels"])
                # labels = self.one_hot(labels)
                # print(labels)
                loss = loss_function(predictions, labels)
                loss.backward()
                optimizer.step()
                # print(predictions)
                label_list = list(set(batch["labels"]))
                predictions_max = torch.argmax(predictions, dim=1)
                acc, recall, prec, f_beta = get_multi_metrics(pred_y=predictions_max, true_y=batch["labels"],
                                                                  labels=label_list)
                current_step += 1
                print("train: step: {}, loss: {}, acc: {}, recall: {}, precision: {}, f_beta: {}".format(
                        current_step, loss, acc, recall, prec, f_beta))

                if current_step % self.config["checkpoint_every"] == 0:
                    eval_losses = []
                    eval_accs = []
                    eval_recalls = []
                    eval_precs = []
                    eval_f_betas = []
                    for eval_batch in self.train_data_obj.next_batch(self.eval_tasks):
                        eval_predictions = self.model(eval_batch)
                        eval_labels = torch.LongTensor(eval_batch["labels"])
                        eval_loss = loss_function(eval_predictions, eval_labels)
                        eval_losses.append(eval_loss)
                        eval_label_list = list(set(eval_batch["labels"]))
                        eval_predictions_max = torch.argmax(eval_predictions, dim=1)
                        acc, recall, prec, f_beta = get_multi_metrics(pred_y=eval_predictions_max,
                                                                    true_y=eval_batch["labels"],
                                                                    labels=eval_label_list)
                        eval_accs.append(acc)
                        eval_recalls.append(recall)
                        eval_precs.append(prec)
                        eval_f_betas.append(f_beta)
                    print("\n")
                    print("eval:  loss: {}, acc: {}, recall: {}, precision: {}, f_beta: {}".format(
                            mean(eval_losses), mean(eval_accs), mean(eval_recalls),
                            mean(eval_precs), mean(eval_f_betas)))
                    print("\n")

                    # if self.config["ckpt_model_path"]:
                    #     save_path = os.path.join(os.path.abspath(os.getcwd()),
                    #                                  self.config["ckpt_model_path"])
                    #     if not os.path.exists(save_path):
                    #         os.makedirs(save_path)
                    #     model_save_path = os.path.join(save_path, self.config["model_name"])
                    #     self.model.saver.save(sess, model_save_path, global_step=current_step)


if __name__ == "__main__":
    config_path = "config.json"
    trainer = PrototypicalTrainer(config_path)
    trainer.train()
