from primitive_perceptron import *

# Different tests:

def test_matrix():
    test_arr = np.array([np.array([1, 3]), np.array([2, 3]), np.array([1]), np.array([4, 5, 6])])
    print(test_arr + test_arr * 0.01)


def test_gen():
    _blues, _reds = generate_test_points(0, 0, 0.6, 0.7, 10000, 2, 2)
    plt.scatter(*zip(*_blues))
    plt.scatter(*zip(*_reds))
    plt.show()

    r = [rand(0, 1) for _ in range(100)]
    print(len([i for i in r if i > 0]), len([i for i in r if i < 0]))


def test_grad_counting():
    data = make_training_data()
    this_perceptron = primitive_perceptron(2, (3, 2, 4), debug=False)

    random_sample = data[0]
    print("Chosen sample:", random_sample)
    err, experimental_grad = this_perceptron.count_experimental_gradient(np.array(random_sample[1:]), random_sample[0])
    print_weights_gradient(experimental_grad)
    err, anal_grad = this_perceptron.count_back_propagational_gradient(np.array(random_sample[1:]), random_sample[0])
    print_weights_gradient(anal_grad)


def show_model_prediction_map(model : primitive_perceptron, number : int):
    blues, reds = generate_test_points(0, 0, 0.645, 0.655, number, 2, 2)
    points = list(blues) + list(reds)
    model_blues = []
    model_reds = []
    for p in points:
        res = model.predict(np.array(p))
        if res:
            model_blues.append(p)
        else:
            model_reds.append(p)


    blue_xs = np.array(model_blues).T[0] if model_blues else []
    blue_ys = np.array(model_blues).T[1] if model_blues else []

    red_xs = np.array(model_reds).T[0] if model_reds else []
    red_ys = np.array(model_reds).T[1] if model_reds else []

    plt.scatter(blue_xs, blue_ys, color="blue")
    plt.scatter(red_xs, red_ys, color="red")

    plt.show()


def test_nn():
    p = primitive_perceptron(2, (40, 20), debug=True)
    losses = []
    for i in range(40):
        print("\n\n____________________________\nReal epoch:", i)
        training_data = make_training_data(1000)
        losses.append(p.SGD_fit([np.array((i[1], i[2])) for i in training_data], [i[0] for i in training_data], 0.1, 10, 1)[0])
        print("Real loss:", losses[-1])
        show_model_prediction_map(model=p, number=10000)
        print("Weights:", p.w)
        # print("max_abs_w:", np.max(np.abs(p.w)))

    print("Losses:", np.array(losses))

    plt.plot(list(range(len(losses))), losses)
    plt.show()


    """
    print(np.array(p.predict(np.array([0.1, -0.5]), True, True)))

    print(
        np.array(p.train([np.array((i[1], i[2])) for i in training_data], [i[0] for i in training_data], 10, 1, True)))
    """



if __name__ == '__main__':
    test_nn()

