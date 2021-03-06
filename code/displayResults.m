function ok = displayResults(Y,Yp,testing,displayPlots)
    error = Y-Yp;
    error_norm = error./(Y+1);
    
    fprintf('\tdz\t\tmax(|z|)\tdz_norm\t\tmax(|z_norm|)\n');

    dz = sqrt(mean(error(testing).^2));
    mz = max(abs(error(testing)));
    dz_norm = sqrt(mean(error_norm(testing).^2));
    mz_norm = max(abs(error_norm(testing)));

    fprintf('\t%f\t%f\t%f\t%f\n',dz,mz,dz_norm,mz_norm);

    if(displayPlots)
        
        dscatter(Y(testing),Yp(testing));
        a = axis;
        hold on;
        plot([a(1);a(2)],[a(1);a(2)],'k--','LineWidth',2);

        xlabel('$z_{s}$','FontSize',30,'Interpreter','LaTex');
        ylabel('$z_{p}$','FontSize',30,'Interpreter','LaTex');
        
        set(gca,'FontSize',16,'FontName','Times');
        axis(a);

        range = max(abs(error(testing)));
        
        figure;
        subplot(1,5,[1 2 3 4]);
        groups = round(20*Y)/20;
        groupID = sort(unique(groups(testing)));

        boxplot(error(testing),groups(testing),'symbol','');
        xlabel('$z_{s}$','FontSize',30,'Interpreter','LaTex');
        ylabel('$z_{s}-z_{p}$','FontSize',30,'Interpreter','LaTex');
        pos1 = get(gca,'Position');

        delete(findall(gcf, 'type', 'text'));
        set(gca, 'xticklabelmode', 'auto', 'xtickmode', 'auto');
        labels = get(gca,'Xtick');
        labels = groupID(labels);
        set(gca,'FontSize',16,'FontName','Times','XtickLabel',labels);

        a = axis;
        axis([a(1) a(2) -range range]);
        subplot(1,5,5);
        pos2 = get(gca,'Position');

        set(gca,'FontSize',16,'FontName','Times','Position',[pos2(1) pos1(2) pos2(3) pos1(4)]);

        histfit(error(testing),50,'kernel');
        h = findobj(gca,'Type','patch');
        set(h,'FaceColor','w','EdgeColor','b');
        set(gca,'CameraUpVector',[-1,0,0]);
        set(gca,'Xdir','reverse');
        set(gca, 'YTick',[]);
        set(gca, 'XTick',[]);
        a = axis;
        axis([-range range a(3) a(4)]);
    end

end