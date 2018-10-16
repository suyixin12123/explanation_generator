Double encoder VAE.

This is the middle part. The image will be feed into two encoder respectively. Both two outputed representation will be using KL divergence with N(0,1). Then two representations will be concatinated together to get the final result. 
