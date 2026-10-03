#include "layers.h"
namespace vocal {
T Layers::flow(T z,T mask,T speaker) {
    for(int i=c.flow_blocks-1;i>=0;--i) {
        z=slice(z,1,c.latent-1,c.latent,-1);
        auto p="flow.flows."+std::to_string(i*2);
        auto a=slice(z,1,0,c.latent/2),b=slice(z,1,c.latent/2,c.latent/2);
        auto x=mul(conv(a,p+".pre"),mask);
        auto cond=conv(speaker,p+".enc.cond_layer");
        T total=nullptr;
        for(int j=0;j<c.flow_layers;j++) {
            auto s=std::to_string(j);
            auto gates=add(conv(x,p+".enc.in_layers."+s,1,c.flow_dilations[j]),slice(cond,1,j*2*c.hidden,2*c.hidden));
            auto y=mul(act(slice(gates,1,0,c.hidden),ActivationType::kTANH),act(slice(gates,1,c.hidden,c.hidden),ActivationType::kSIGMOID));
            y=conv(y,p+".enc.res_skip_layers."+s);
            if(j+1<c.flow_layers) { x=mul(add(x,slice(y,1,0,c.hidden)),mask);y=slice(y,1,c.hidden,c.hidden); }
            total=total?add(total,y):y;
        }
        auto mean=mul(conv(mul(total,mask),p+".post"),mask);
        z=concat({a,mul(sub(b,mean),mask)},1);
    }
    z=mul(z,mask);if(o.debug_outputs)mark(z,"flow_z");return z;
}
}
