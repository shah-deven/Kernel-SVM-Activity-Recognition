load('q3_1_data.mat');
lid = trLb(:,1) == -1;
trLb(lid, 1) = 2;
lid = valLb(:,1) == -1;
valLb(lid, 1) = 2;

epochs = 2000;
ita_0 = 1;
ita_1 = 100;
c = 10;
k = 2;
x = trD;
[d, n] = size(trD);
w = zeros(d, k);
y = trLb;
losses = zeros(1, epochs);
for epoch = 1:epochs
    ita = ita_0 / (ita_1 + epoch);
    new_numbers = randperm(n);
    loss = 0;
    w_transpose = w';
    for i = new_numbers(:)' % as per instructions
        weight_sum = 0;
        y_i_hat = inf;
        y_i_hat_value = -inf;
        for j = 1:k % to calculate y_i_hat
            if j ~= y(i)
                temp = w_transpose(j, :) * x(:, i);
                if temp > y_i_hat_value
                    y_i_hat_value = temp;
                    y_i_hat = j;
                end
            end
        end
        loss_value = w_transpose(y_i_hat, :) * x(:, i) - w_transpose(y(i), :) * x(:, i) + 1;
        for orig_j = 1:k % for updating all wj
            w_transpose = w';
            if loss_value > 0
                if orig_j == y(i)
                    w(:, orig_j) = w(:, orig_j) - ita * ((w(:, orig_j) / n) - c * x(:, i));
                elseif orig_j == y_i_hat
                    w(:, orig_j) = w(:, orig_j) - ita * ((w(:, orig_j) / n) + c * x(:, i));
                else
                    w(:, orig_j) = w(:, orig_j) - ita * (w(:, orig_j) / n);
                end
            else
                w(:, orig_j) = w(:, orig_j) - ita * (w(:, orig_j) / n);
            end
            weight_sum = weight_sum + (norm(w(:, orig_j)) ^ 2);
        end
        weight_sum = weight_sum / (2 * n);
        temp_loss = c * max(loss_value, 0);
        loss = loss + weight_sum + temp_loss;
    end
    losses(epoch) = loss;
end
numbers = linspace(1, epochs, epochs);
plot(numbers, losses);
title("Loss after each epoch");
xlabel("Epochs");
ylabel("Loss value");


[d, k] = size(w);

to_check = zeros(1, 2);
y_pred = zeros(1, vn);
for i = 1:n
    for j = 1:k
        to_check(j) = w(:, j)' * trD(:, i);
    end
    [temp_value, temp_index] = max(to_check);
    y_pred(i) = temp_index;
end
y_pred = y_pred';
correct_predictions = 0;
wrong_predictions = 0;

for i = 1:n
    if y_pred(i) == trLb(i)
        correct_predictions = correct_predictions + 1;
    else
        wrong_predictions = wrong_predictions + 1;
    end
end

disp("prediction error on trD: ");
disp(wrong_predictions / n);


[d, k] = size(w);
[d, vn] = size(valD);

to_check = zeros(1, 2);
y_pred = zeros(1, vn);
for i = 1:vn
    for j = 1:k
        to_check(j) = w(:, j)' * valD(:, i);
    end
    [temp_value, temp_index] = max(to_check);
    y_pred(i) = temp_index;
end
y_pred = y_pred';
correct_predictions = 0;
wrong_predictions = 0;

for i = 1:vn
    if y_pred(i) == valLb(i)
        correct_predictions = correct_predictions + 1;
    else
        wrong_predictions = wrong_predictions + 1;
    end
end

disp("prediction error on valD: ");
disp(wrong_predictions / vn);
disp("objective function value: ");
disp(losses(2000));
disp("l2 norm of weights and its square");
disp(norm(w) ^ 2);
