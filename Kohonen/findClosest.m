function label = findClosest(proto, data, labels)
    closest_dist = sum((proto-data(1,:)).^2); % distance initialization
    for i=1:size(data,1)
        if sum((proto-data(i,:)).^2) < closest_dist
            closest_dist = sum((proto-data(i,:)).^2);
            label = labels(i);
        end
    end
end