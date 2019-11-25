import random


class RandomSelector:

    def select_next_batch(self, model, training_set, selection_count):
        scores = []
        for i in range(len(training_set.remaining_image_paths)):
            scores.append(random.random())
        selected_samples = list(zip(*sorted(zip(scores, training_set.remaining_image_paths), key=lambda x: x[0], reverse=True)))[1][:selection_count]
        training_set.expand_training_set(selected_samples)
