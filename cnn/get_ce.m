function val = get_ce(z,act_fun)
switch act_fun
    case 'logistic'
        val = (z).*(1-z);
    case 'tanh'
        val = 1-z.^2;
    case 'relu'
        val = z>0;
end