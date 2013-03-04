% xlsread : reads file into matrix
a = xlsread('data.xls');
b = zeros(650,5);
for i = 1 : 3
  minvalue = min(a(:,i));
  maxvalue = max(a(:,i));
    for j = 1 : 650
       b(j,i) = (2* a(j, i) - (minvalue + maxvalue)) / (maxvalue - minvalue);   
        
    end
end

for i = 4 : 5
  minvalue = min(a(:,i));
  maxvalue = max(a(:,i));
    for j = 1 : 650
       b(j,i) = (a(j, i) - minvalue) / (maxvalue - minvalue);
    end
end

xlswrite('data2.xls', b);