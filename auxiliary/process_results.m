function [ t_acc ll micro_F h_acc ] = process_results( Obj )

for i=1:length(Obj)
    t_acc(i)=Obj{i}.ExactMatch;
    h_acc(i)=Obj{i}.HammingMatch;
    if(isfield(Obj{i},'ll'))
        ll(i)=Obj{i}.ll;
    else
        ll(i)=0;
    end
    micro_F(i)=Obj{i}.MicroF1;
    macro_F(i)=Obj{i}.MacroF1;
end


fprintf( 'EMA = %f\n', mean(t_acc));
fprintf( 'LL_ts = %f\n', mean(ll));
fprintf( 'microF1 = %f\n', mean(micro_F));
fprintf( 'macroF1 = %f\n', mean(macro_F));
fprintf( 'EMA/round = \n\t' );
fprintf( '%f    ', t_acc );
fprintf( '\n' );

end

