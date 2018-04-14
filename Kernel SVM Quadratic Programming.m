%lid = trLb(:,1) == -1;
%trLb(lid, 1) = 2;
load('q3_1_data.mat');
c = 0.1;
if c == 0.1
    epsilon = 0.00001;
elseif c == 10
    epsilon = 0.1;
end
x = trD;
y = trLb;

[d, n] = size(trD);
f = ones(n, 1);
f = -1 * f;
h = zeros(n, n);

for i = 1:n
    for j = 1:n
        h(i, j) = dot(x(:, i), x(:, j)) * y(i) * y(j);
    end
end

A = [];
b = [];
A_eq = trLb';
b_eq = 0;
lb = zeros(n, 1);
ub = c * ones(n, 1);
[alpha, f_val] = quadprog(h, f, A, b, A_eq, b_eq, lb, ub);

%disp(f_val);
temp = y .* alpha;
w = x * temp;

temp = abs(alpha - 0.05);
[alpha_min, index] = min(temp);
bias = y(index) - (w' * x(:, index));

y_pred = (w' * valD) + bias;
for i = 1:size(y_pred, 2)
    if y_pred(i) < 0
        y_pred(i) = -1;
    else
        y_pred(i) = 1;
    end
end

correct_predictions = 0;
wrong_predictions = 0;
true_positive = 0;
false_positive = 0;
true_negative = 0;
false_negative = 0;
for i = 1:size(y_pred, 2)
    if y_pred(i) == valLb(i)
        if y_pred(i) == 1
            true_positive = true_positive + 1;
        elseif y_pred(i) == -1
            true_negative = true_negative + 1;
        end
        correct_predictions = correct_predictions + 1;
    else
        if y_pred(i) == 1
            false_positive = false_positive + 1;
        elseif y_pred(i) == -1
            false_negative = false_negative + 1;
        end
        wrong_predictions = wrong_predictions + 1;
    end
end
disp("accuracy: ");
disp(correct_predictions * 100 / (correct_predictions + wrong_predictions));
disp("objective function value: ");
disp(-1 * f_val);
%disp(wrong_predictions);
disp("confusion matrix : ");
disp(sprintf("%d %d", true_positive, false_positive));
disp(sprintf("%d %d", false_negative, true_negative));

support_vectors_count = 0;
for i = 1:length(alpha)
    if alpha(i) > epsilon
        support_vectors_count = support_vectors_count + 1;
    end
end

disp("Number of support vectors: ");
disp(support_vectors_count);
