function  Plotb5n3(variable,STD,CRLB53,dim)
switch dim
    case 3
        vname='z(nm)';
    case 4
        vname='\phi(\circ)';
    case 5
        vname='\theta(\circ)';
    case 6 
        vname='g_2';
        variable(1)=0;
end
sztx=20;
figure('name','Orientation CRLB','NumberTitle','off');
hold on;
plot(variable,CRLB53(:,[6]),'Color','#0072BD','LineWidth',2);
plot(variable,CRLB53(:,[7]),'Color','#D95319','LineWidth',2);
scatter(variable,STD(:,6),'filled','b','MarkerFaceColor','#0072BD')
scatter(variable,STD(:,7),'filled','r','MarkerFaceColor','#D95319')
legend({'CRLB_\phi^{1/2}','CRLB_\theta^{1/2}','\sigma_\phi','\sigma_\theta'},'NumColumns',2);
xlabel(vname);
ylabel('Precision(\circ)')
xlim([variable(1) variable(end)]);ylim([0 10])
set(gca,'FontSize',sztx,'FontWeight','bold','XTick',linspace(variable(1),variable(end),5));

figure('name','position CRLB','NumberTitle','off');
hold on;
plot(variable,CRLB53(:,1),'Color','#0072BD','LineWidth',2);
scatter(variable,STD(:,1),'filled','b','MarkerFaceColor','#0072BD')
plot(variable,CRLB53(:,2),'Color','#D95319','LineWidth',2);
scatter(variable,STD(:,2),'filled','r','MarkerFaceColor','#D95319')
plot(variable,CRLB53(:,5),'Color','#EDB120','LineWidth',2);
scatter(variable,STD(:,5),'filled','MarkerFaceColor','#EDB120')
xlabel(vname);
legend({'CRLB_x^{1/2}','\sigma_x','CRLB_y^{1/2}','\sigma_y','CRLB_z^{1/2}','\sigma_z'},'NumColumns',3);
ylabel('Precision(nm)')
xlim([variable(1) variable(end)]);ylim([0 30])
set(gca,'FontSize',sztx,'FontWeight','bold','XTick',linspace(variable(1),variable(end),5));

figure('name','g2 CRLB','NumberTitle','off');plot(variable,CRLB53(:,8),'Color','#0072BD','LineWidth',2);
hold on;
scatter(variable,STD(:,8),'filled','b','MarkerFaceColor','#0072BD')
xlim([variable(1) variable(end)])

legend('CRLB_{g2}^{1/2}','\sigma_{g2}')
ylim([0 0.2])
xlabel(vname);
set(gca,'FontSize',sztx,'FontWeight','bold','XTick',linspace(variable(1),variable(end),5));

end

