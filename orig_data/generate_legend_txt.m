function text=generate_legend_txt(selected_buses_plusone)
num_bus_display = length(selected_buses_plusone);
text = cell(num_bus_display,1);
for ll = 1:num_bus_display
    text{ll} = ['bus ',int2str(selected_buses_plusone(ll))];
end