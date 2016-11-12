require './neural_network'

LOCATIONS = [:home, :work, :tennis, :parents]

LOCATIONS_INDEXED = LOCATIONS.map.with_index { |x, i| [x, i] }.to_h

XX = [
  # week 1
  # 1st day of week, 8am
  [:work, 1, 8], [:tennis, 1, 17], [:home, 1, 20],
  [:work, 2, 8], [:home,   2, 18],
  [:work, 3, 8], [:tennis, 3, 17], [:home, 3, 20],
  [:work, 4, 8], [:home,   4, 18],
  [:work, 5, 8], [:home,   5, 18],

  [:parents, 7, 13], [:home, 7, 18],

  # week 2
  [:work, 1, 8], [:home, 1, 18],
  [:work, 2, 8], [:home, 2, 18],
  [:work, 3, 8], [:tennis, 3, 17], [:home, 3, 20],
  [:work, 4, 8], [:home, 4, 18],
  [:work, 5, 8], [:home, 5, 18],

  # week 3

  [:work, 1, 8],
  [:work, 2, 8],
  [:work, 3, 8],
  [:work, 4, 8],
  [:work, 5, 8],

  [:tennis, 1, 17],
  [:home, 1, 20],
  [:home, 2, 18],
  [:tennis, 3, 17],
  [:home, 3, 20],
  [:home, 4, 18],
  [:home, 5, 18],

  [:parents, 7, 12],

  # week 4
  [:work, 1, 8],
  [:work, 2, 8],
  [:work, 3, 8],
  [:work, 4, 8],
  [:work, 5, 8],

  [:home, 1, 18],
  [:home, 2, 18],
  [:tennis, 3, 17],
  [:home, 3, 20],
  [:home, 4, 18],
  [:home, 5, 18],
]

xx = []
yy = []

XX.each do |destination, day, time|
  yy << LOCATIONS_INDEXED[destination]
  xx << [day.to_f/7, time.to_f/24]
end

nn = NeuralNetwork.new
nn.train_with_probabilities(xx, yy, LOCATIONS.size, [25], 400, 1, 0.001)

res = nn.predict_with_probabilities(xx)

yy_res = res.collect(&:first).map {|v| v[1]}

found = NeuralNetwork.compare_results(yy, yy_res)
puts "NN accuracy=#{found.to_f/xx.size}"

[
  [1, 16.5], [1, 17], [1, 17.5], [1, 17.8],
  [2, 17], [2, 18.1],
  [4, 18],
  [6, 23],
  [7, 13]
].each do |day, time|
  res = nn.predict_with_probabilities([
    [day.to_f/7, time.to_f/24]
  ]).first.select {|v| v[0] > 0}
  puts "#{day} #{time} \t #{res.map {|v| [LOCATIONS[v[1]], v[0]]}.inspect}"
end


# puts nn.predict_with_probabilities([[1.to_f/7, 16.5.to_f/24]]).first.map {|v| [LOCATIONS[v[1]], v[0]]}.inspect
# puts nn.predict_with_probabilities([[1.to_f/7, 17.to_f/24]]).first.map {|v| [LOCATIONS[v[1]], v[0]]}.inspect
# puts nn.predict_with_probabilities([[1.to_f/7, 17.5.to_f/24]]).first.map {|v| [LOCATIONS[v[1]], v[0]]}.inspect
# puts nn.predict_with_probabilities([[1.to_f/7, 17.75.to_f/24]]).first.map {|v| [LOCATIONS[v[1]], v[0]]}.inspect
# puts nn.predict_with_probabilities([[2.to_f/7, 18.to_f/24]]).first.map {|v| [LOCATIONS[v[1]], v[0]]}.inspect
# puts nn.predict_with_probabilities([[4.to_f/7, 18.to_f/24]]).first.map {|v| [LOCATIONS[v[1]], v[0]]}.inspect
# puts nn.predict_with_probabilities([[7.to_f/7, 13.to_f/24]]).first.map {|v| [LOCATIONS[v[1]], v[0]]}.inspect
# puts nn.predict_with_probabilities([[2.to_f/7, 17.to_f/24]]).first.map {|v| [LOCATIONS[v[1]], v[0]]}.inspect
# puts nn.predict_with_probabilities([[6.to_f/7, 23.to_f/24]]).first.map {|v| [LOCATIONS[v[1]], v[0]]}.inspect
