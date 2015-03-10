function T = generate_random_linear_chain(d)

T = cell(1,d);
tmp = randperm(d);

for i = 1:d
    T{i}.node = tmp(i);

    if i == 1
        T{i}.parent = [];
    else
        T{i}.parent = tmp(i-1);
    end
    if i == d
        T{i}.children = [];
    else
        T{i}.children = tmp(i+1);
    end
end

end