require 'ruby-fann'

class NeuralNetwork
  attr_reader :examples_count
  attr_reader :features_count
  attr_reader :hidden_neurons
  attr_reader :outputs_count
  attr_reader :max_epochs
  attr_reader :epochs_between_reports
  attr_reader :desired_error

  def train(xx, yy, hidden_neurons, max_epochs, epochs_between_reports, desired_error)
    raise "wrong args" unless xx.size == yy.size
    @hidden_neurons, @max_epochs, @epochs_between_reports, @desired_error = hidden_neurons, max_epochs, epochs_between_reports, desired_error
    @examples_count = xx.size
    @features_count = xx[0].size
    @outputs_count = yy[0].size

    train = RubyFann::TrainData.new(inputs: xx, desired_outputs: yy)
    train.shuffle
    @fann = RubyFann::Standard.new(num_inputs: @features_count, hidden_neurons: hidden_neurons, num_outputs: @outputs_count)
    @fann.train_on_data(train, max_epochs, epochs_between_reports, desired_error)
  end

  def train_with_probabilities(xx, labels, outputs, hidden_neurons, max_epochs, epochs_between_reports, desired_error)
    yy = []
    labels.each do |l|
      y = Array.new(outputs, 0)
      y[l] = 1
      yy << y
    end

    return train(xx, yy, hidden_neurons, max_epochs, epochs_between_reports, desired_error)
  end

  def predict_with_probabilities(xx)
    xx.map do |x|
      res = @fann.run(x)
      res.each_with_index.to_a.sort { |a, b|  b[0] <=> a[0] }
    end
  end

  def predict(xx)
    xx.map do |x|
      res = @fann.run(x)
      res = yield(res) if block_given?
      res
    end
  end

  def self.compare_results(yy, hh)
    res = 0
    yy.each_with_index do |y, i|
      res += 1 if y == hh[i]
    end
    res
  end
end
