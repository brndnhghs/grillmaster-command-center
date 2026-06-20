"""
Post-processing filter module — applies OpenCV effects to generated images.

Usage via pipeline.py:
  --filter oil                        # single effect
  --filter '{"effect":"colormap","colormap":"ocean"}'  # JSON
  --filter '{"effect":"edge","mode":"glow"}'           # multiple params
  --filter '{"effect":"bloom","strength":0.5}'         # parameterized
"""

from __future__ import annotations
from pathlib import Path
import math
import random

import numpy as np
from PIL import Image


def _ensure_cv2():
    import cv2
    return cv2


def _parse_spec(filter_spec):
    if isinstance(filter_spec, str):
        return {"effect": filter_spec}
    return dict(filter_spec)


_NAMED_COLORS = {
    "red": (255,40,40), "green": (40,200,40), "blue": (40,40,255),
    "cyan": (40,255,255), "magenta": (255,40,255), "yellow": (255,255,40),
    "orange": (255,160,40), "purple": (160,40,255), "pink": (255,100,180),
    "teal": (40,180,180), "white": (255,255,255), "black": (0,0,0),
    "cream": (255,240,200), "gold": (255,200,40), "silver": (200,200,200),
    "navy": (20,20,100), "maroon": (100,20,20), "lime": (40,255,40),
    "coral": (255,120,80), "indigo": (75,0,130), "violet": (140,80,200),
    "amber": (255,180,0), "emerald": (40,200,100), "ruby": (200,20,60),
}


def _parse_color(name):
    return _NAMED_COLORS.get(name.lower().strip(), (255,255,255))


def apply_filter(image_path, filter_spec, out_path=None):
    cv2 = _ensure_cv2()
    spec = _parse_spec(filter_spec)
    effect = spec.get("effect", "")
    img = Image.open(image_path).convert("RGB")
    arr = np.array(img, dtype=np.uint8)
    h, w = arr.shape[:2]
    if out_path is None:
        out_path = image_path

    if effect == "none" or effect == "":
        pass
    elif effect == "oil":
        r = int(spec.get("radius",4)); l = int(spec.get("levels",10))
        if hasattr(cv2,"xphoto") and hasattr(cv2.xphoto,"oilPainting"): arr = cv2.xphoto.oilPainting(arr,r,l)
        else: arr = cv2.bilateralFilter(arr,r*2+1,50,50)
    elif effect == "detail":
        s = float(spec.get("sigma",20)); a = float(spec.get("amount",1.0))
        d = cv2.detailEnhance(arr,s,a); arr = cv2.addWeighted(arr,1-min(a,1),d,min(a,1),0) if a<1 else d
    elif effect == "sharpen":
        i = float(spec.get("intensity",1.0))
        k = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]],dtype=np.float32)
        arr = cv2.addWeighted(arr,1-i,cv2.filter2D(arr,-1,k),i,0)
    elif effect == "emboss":
        k = np.array([[-2,-1,0],[-1,1,1],[0,1,2]],dtype=np.float32)
        arr = cv2.addWeighted(arr,1-float(spec.get("alpha",0.4)),cv2.filter2D(arr,-1,k),float(spec.get("alpha",0.4)),0)
    elif effect == "pencil":
        g = cv2.cvtColor(arr,cv2.COLOR_RGB2GRAY)
        p = cv2.divide(g,255-cv2.GaussianBlur(cv2.bitwise_not(g),(21,21),0),scale=256)
        arr = cv2.addWeighted(arr,1-float(spec.get("alpha",0.7)),cv2.cvtColor(p,cv2.COLOR_GRAY2RGB),float(spec.get("alpha",0.7)),0)
    elif effect == "edge":
        m = spec.get("mode","overlay"); e = cv2.Canny(arr,int(spec.get("canny_low",30)),int(spec.get("canny_high",100)))
        if m=="mono": arr=cv2.cvtColor(e,cv2.COLOR_GRAY2RGB)
        elif m=="glow":
            eb=cv2.GaussianBlur(e.astype(np.float32),(0,0),sigmaX=3,sigmaY=3)
            ec=np.stack([eb*0.8,eb*0.3,eb*0.1],axis=-1).clip(0,255).astype(np.uint8)
            arr=cv2.addWeighted(arr,1-float(spec.get("alpha",0.6)),ec,float(spec.get("alpha",0.6)),0)
        else: arr=cv2.addWeighted(arr,1-float(spec.get("alpha",0.3)),cv2.cvtColor(e,cv2.COLOR_GRAY2RGB)*255,float(spec.get("alpha",0.3)),0)
    elif effect == "colormap":
        cm = spec.get("colormap","jet")
        mp = {"autumn":cv2.COLORMAP_AUTUMN,"bone":cv2.COLORMAP_BONE,"jet":cv2.COLORMAP_JET,"winter":cv2.COLORMAP_WINTER,"rainbow":cv2.COLORMAP_RAINBOW,"ocean":cv2.COLORMAP_OCEAN,"summer":cv2.COLORMAP_SUMMER,"spring":cv2.COLORMAP_SPRING,"cool":cv2.COLORMAP_COOL,"hsv":cv2.COLORMAP_HSV,"pink":cv2.COLORMAP_PINK,"hot":cv2.COLORMAP_HOT,"parula":cv2.COLORMAP_PARULA,"magma":cv2.COLORMAP_MAGMA,"inferno":cv2.COLORMAP_INFERNO,"plasma":cv2.COLORMAP_PLASMA,"viridis":cv2.COLORMAP_VIRIDIS,"cividis":cv2.COLORMAP_CIVIDIS,"twilight":cv2.COLORMAP_TWILIGHT,"turbo":cv2.COLORMAP_TURBO,"deepgreen":cv2.COLORMAP_DEEPGREEN}
        c=cv2.applyColorMap(cv2.cvtColor(arr,cv2.COLOR_RGB2GRAY),mp.get(cm,cv2.COLORMAP_JET))
        cr=cv2.cvtColor(c,cv2.COLOR_BGR2RGB)
        a=float(spec.get("alpha",1.0))
        arr=cv2.addWeighted(arr,1-a,cr,a,0) if a<1 else cr
    elif effect == "chroma":
        s=int(spec.get("shift",5)); d=spec.get("direction","h")
        r,g,b=cv2.split(arr)
        if d=="h": rs=np.roll(r,s,1);bs=np.roll(b,-s,1)
        else: rs=np.roll(r,s,0);bs=np.roll(b,-s,0)
        arr=cv2.merge([rs,g,bs])
    elif effect == "morph":
        o=spec.get("op","dilate"); k=int(spec.get("kernel",3)); it=int(spec.get("iterations",1))
        om={"dilate":cv2.MORPH_DILATE,"erode":cv2.MORPH_ERODE,"open":cv2.MORPH_OPEN,"close":cv2.MORPH_CLOSE,"gradient":cv2.MORPH_GRADIENT,"tophat":cv2.MORPH_TOPHAT,"blackhat":cv2.MORPH_BLACKHAT}
        arr=cv2.cvtColor(cv2.morphologyEx(cv2.cvtColor(arr,cv2.COLOR_RGB2GRAY),om.get(o,cv2.MORPH_DILATE),np.ones((k,k),dtype=np.uint8),iterations=it),cv2.COLOR_GRAY2RGB)
    elif effect == "cartoon":
        arr=cv2.stylization(arr,sigma_s=float(spec.get("sigma_s",60)),sigma_r=float(spec.get("sigma_r",0.5)))
    elif effect == "clahe":
        lab=cv2.cvtColor(arr,cv2.COLOR_RGB2LAB); l,a,b=cv2.split(lab)
        c=cv2.createCLAHE(clipLimit=float(spec.get("clip_limit",2)),tileGridSize=(int(spec.get("grid_size",8)),)*2)
        arr=cv2.cvtColor(cv2.merge([c.apply(l),a,b]),cv2.COLOR_LAB2RGB)
    elif effect == "vignette":
        st=float(spec.get("strength",1.0)); yy,xx=np.mgrid[:h,:w]; cx,cy=w/2,h/2
        v=(1.0-np.sqrt((xx-cx)**2+(yy-cy)**2)/np.sqrt(cx**2+cy**2)*st).clip(0,1)
        for c in range(3): arr[:,:,c]=(arr[:,:,c].astype(np.float32)*v).clip(0,255).astype(np.uint8)
    elif effect == "bloom":
        th=int(spec.get("threshold",200)); bs=float(spec.get("blur_sigma",10)); a=float(spec.get("alpha",0.4))
        m=(cv2.cvtColor(arr,cv2.COLOR_RGB2GRAY)>th).astype(np.uint8)*255
        if m.sum()>0:
            b=cv2.GaussianBlur(arr.astype(np.float32),(0,0),sigmaX=bs)
            bm=cv2.GaussianBlur(m.astype(np.float32),(0,0),sigmaX=bs//2)[:,:,None]/255
            arr=(arr.astype(np.float32)+b*bm*a).clip(0,255).astype(np.uint8)
    elif effect == "pixelate":
        b=int(spec.get("block",16))
        arr=cv2.resize(cv2.resize(arr,(w//b,h//b),interpolation=cv2.INTER_LINEAR),(w,h),interpolation=cv2.INTER_NEAREST)
    elif effect == "blur":
        t=spec.get("type","gaussian"); k=int(spec.get("ksize",15)); s=float(spec.get("sigma",0))
        if k%2==0: k+=1
        if t=="gaussian": arr=cv2.GaussianBlur(arr,(k,k),s)
        elif t=="median": arr=cv2.medianBlur(arr,min(k,49))
        elif t=="bilateral": arr=cv2.bilateralFilter(arr,min(k,49),50,50)
        elif t=="stack":
            for _ in range(int(spec.get("passes",3))): arr=cv2.GaussianBlur(arr,(0,0),sigmaX=s)
    elif effect == "palette":
        from .utils import quantize_to_palette
        arr=(quantize_to_palette(arr.astype(np.float32)/255,spec.get("palette","pico8"))*255).astype(np.uint8)
    elif effect == "swirl":
        c=spec.get("center",None); st=float(spec.get("strength",1.0)); r=float(spec.get("radius",min(w,h)*0.5))
        cx,cy=c if c else (w/2,h/2)
        yy,xx=np.mgrid[:h,:w].astype(np.float32); dx,dy=xx-cx,yy-cy
        a=st*np.exp(-np.sqrt(dx**2+dy**2)/r)
        arr=cv2.remap(arr,np.clip(cx+dx*np.cos(a)-dy*np.sin(a),0,w-1).astype(np.float32),np.clip(cy+dx*np.sin(a)+dy*np.cos(a),0,h-1).astype(np.float32),cv2.INTER_LINEAR)
    elif effect == "twist":
        a=float(spec.get("amplitude",20)); f=float(spec.get("frequency",0.05)); d=spec.get("direction","h")
        yy,xx=np.mgrid[:h,:w].astype(np.float32)
        if d=="h": arr=cv2.remap(arr,np.clip(xx+a*np.sin(yy*f),0,w-1).astype(np.float32),yy.astype(np.float32),cv2.INTER_LINEAR)
        else: arr=cv2.remap(arr,xx.astype(np.float32),np.clip(yy+a*np.sin(xx*f),0,h-1).astype(np.float32),cv2.INTER_LINEAR)
    elif effect == "ripple":
        a=float(spec.get("amplitude",10)); f=float(spec.get("frequency",0.1))
        yy,xx=np.mgrid[:h,:w].astype(np.float32); d=np.sqrt((xx-w/2)**2+(yy-h/2)**2)
        rp=a*np.sin(d*f)
        arr=cv2.remap(arr,np.clip(xx+rp*(xx-w/2)/(d+1e-6),0,w-1).astype(np.float32),np.clip(yy+rp*(yy-h/2)/(d+1e-6),0,h-1).astype(np.float32),cv2.INTER_LINEAR)
    elif effect == "bulge":
        s=float(spec.get("strength",0.5)); r=float(spec.get("radius",min(w,h)*0.4)); cx,cy=w/2,h/2
        yy,xx=np.mgrid[:h,:w].astype(np.float32); dx,dy=xx-cx,yy-cy; d=np.sqrt(dx**2+dy**2)
        sc=np.where(d<r,1-s*d/r,1.0)
        arr=cv2.remap(arr,np.clip(cx+dx/sc,0,w-1).astype(np.float32),np.clip(cy+dy/sc,0,h-1).astype(np.float32),cv2.INTER_LINEAR)
    elif effect == "pinch":
        s=float(spec.get("strength",0.5)); r=float(spec.get("radius",min(w,h)*0.4)); cx,cy=w/2,h/2
        yy,xx=np.mgrid[:h,:w].astype(np.float32); dx,dy=xx-cx,yy-cy; d=np.sqrt(dx**2+dy**2)
        sc=np.where(d<r,1+s*d/r,1.0)
        arr=cv2.remap(arr,np.clip(cx+dx/sc,0,w-1).astype(np.float32),np.clip(cy+dy/sc,0,h-1).astype(np.float32),cv2.INTER_LINEAR)
    elif effect == "kaleidoscope":
        s=int(spec.get("segments",6)); cx,cy=w//2,h//2
        yy,xx=np.mgrid[:h,:w].astype(np.float32); dx,dy=xx-cx,yy-cy; a=np.arctan2(dy,dx); r=np.sqrt(dx**2+dy**2)
        sa=2*math.pi/s; a=a%sa; a=np.where(a>sa/2,sa-a,a)
        arr=cv2.remap(arr,np.clip(cx+r*np.cos(a),0,w-1).astype(np.float32),np.clip(cy+r*np.sin(a),0,h-1).astype(np.float32),cv2.INTER_LINEAR)

    elif effect == "waves":
        a=float(spec.get("amplitude",15)); fx=float(spec.get("freq_x",0.03)); fy=float(spec.get("freq_y",0.03))
        yy,xx=np.mgrid[:h,:w].astype(np.float32)
        src_x=np.clip(xx+a*np.sin(yy*fx),0,w-1).astype(np.float32)
        src_y=np.clip(yy+a*np.sin(xx*fy),0,h-1).astype(np.float32)
        arr=cv2.remap(arr,src_x,src_y,cv2.INTER_LINEAR)
    elif effect == "sphere":
        cx,cy=w/2,h/2; r=float(spec.get("radius",min(w,h)*0.45))
        yy,xx=np.mgrid[:h,:w].astype(np.float32); dx,dy=(xx-cx)/r,(yy-cy)/r; d=np.sqrt(dx**2+dy**2)
        m=d<1; z=np.sqrt(np.clip(1-d**2,0,None))
        nx=np.divide(dx,z,out=np.zeros_like(dx),where=z>1e-8); ny=np.divide(dy,z,out=np.zeros_like(dy),where=z>1e-8)
        sx=np.clip(cx+nx/(2*np.sqrt(np.clip(0.5+0.5*z,1e-8,None)))*r,0,w-1).astype(np.float32)
        sy=np.clip(cy+ny/(2*np.sqrt(np.clip(0.5+0.5*z,1e-8,None)))*r,0,h-1).astype(np.float32)
        arr=cv2.remap(arr,sx,sy,cv2.INTER_LINEAR); arr[~m]=0
    elif effect == "watercolor":
        r=int(spec.get("radius",5)); r+=1 if r%2==0 else 0
        arr=cv2.medianBlur(arr,min(r,49)); arr=cv2.bilateralFilter(arr,min(r,49),50,50)
        if spec.get("edges",False):
            e=cv2.adaptiveThreshold(cv2.cvtColor(arr,cv2.COLOR_RGB2GRAY),255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,11,2)
            arr=cv2.addWeighted(arr,0.85,cv2.cvtColor(e,cv2.COLOR_GRAY2RGB),0.15,0)
    elif effect == "sketch":
        b=int(spec.get("blur",21)); b+=1 if b%2==0 else 0
        g=cv2.cvtColor(arr,cv2.COLOR_RGB2GRAY); i=cv2.bitwise_not(g)
        sk=cv2.divide(g,255-cv2.GaussianBlur(i,(b,b),0),scale=256)
        if spec.get("colored",False): arr=(arr.astype(np.float32)/255*(cv2.cvtColor(sk,cv2.COLOR_GRAY2RGB).astype(np.float32)/255)*255).clip(0,255).astype(np.uint8)
        else: arr=cv2.cvtColor(sk,cv2.COLOR_GRAY2RGB)
    elif effect == "neon":
        b=int(spec.get("blur",5)); b+=1 if b%2==0 else 0; i=float(spec.get("intensity",1.5))
        e=cv2.Canny(arr,30,100); e=cv2.dilate(e,np.ones((3,3),dtype=np.uint8),iterations=1)
        g=cv2.GaussianBlur(e.astype(np.float32),(b,b),0)/255*i
        r=arr.astype(np.float32)/255; m=e>0
        for c in range(3): ch=r[:,:,c].copy(); ch[m]=np.clip(ch[m]*(1+g[m]),0,1); r[:,:,c]=ch+g*0.3
        arr=(r.clip(0,1)*255).astype(np.uint8)
    elif effect == "glitch":
        i=float(spec.get("intensity",0.5)); r,g,b=cv2.split(arr); h2,w2=r.shape
        for _ in range(max(1,int(i*20))):
            y=random.randint(0,h2-5); sh=random.randint(2,max(3,int(h2*i*0.2)))
            s=random.randint(-int(w2*0.1*i),int(w2*0.1*i)); t={"r":r,"g":g,"b":b}[random.choice(["r","g","b"])]
            if s>0: t[y:y+sh,s:]=t[y:y+sh,:-s].copy(); t[y:y+sh,:s]=0
            elif s<0: t[y:y+sh,:s]=t[y:y+sh,-s:].copy(); t[y:y+sh,s:]=0
        if spec.get("scanlines",False):
            for i2 in range(0,h2,3): r[i2:i2+1]=(r[i2:i2+1].astype(np.float32)*0.6).astype(np.uint8); g[i2:i2+1]=g[i2:i2+1]*0.6; b[i2:i2+1]=b[i2:i2+1]*0.6
        arr=cv2.merge([r,g,b])
    elif effect == "halftone":
        cs=int(spec.get("cell_size",8)); g=cv2.cvtColor(arr,cv2.COLOR_RGB2GRAY)
        r=np.ones(g.shape,dtype=np.uint8)*255
        for y in range(0,h,cs):
            for x in range(0,w,cs):
                cv2.circle(r,(x+cs//2,y+cs//2),max(1,int((1-g[y:y+cs,x:x+cs].mean()/255)*cs/2)),0,-1)
        arr=cv2.cvtColor(r,cv2.COLOR_GRAY2RGB)
    elif effect == "mosaic":
        cs=int(spec.get("cell_size",20))
        for y in range(0,h,cs):
            for x in range(0,w,cs):
                arr[y:y+cs,x:x+cs]=arr[y:y+cs,x:x+cs].mean(axis=(0,1)).astype(np.uint8)
    elif effect == "stained_glass":
        cs=int(spec.get("cell_size",15)); random.seed(int(spec.get("seed",42)))
        pts=[(np.clip(gx*cs+random.randint(-cs//3,cs//3),0,w-1),np.clip(gy*cs+random.randint(-cs//3,cs//3),0,h-1)) for gy in range(h//cs+2) for gx in range(w//cs+2)]
        yy,xx=np.mgrid[:h,:w]; dists=np.zeros((h,w,len(pts)))
        for i,(px,py) in enumerate(pts): dists[:,:,i]=(xx-px)**2+(yy-py)**2
        n=np.argmin(dists,axis=2); r=np.zeros_like(arr)
        for i,(px,py) in enumerate(pts): m=n==i; r[m]=arr[py,px] if m.any() else r[m]
        e=cv2.dilate(cv2.Canny(cv2.cvtColor(r,cv2.COLOR_RGB2GRAY),10,50),np.ones((2,2),dtype=np.uint8),iterations=1)
        r[e>0]=(30,30,30); arr=r
    elif effect == "posterize":
        l=int(spec.get("levels",4)); step=255//(l-1)
        arr=(arr.astype(np.float32)//step*step+step//2).clip(0,255).astype(np.uint8)
    elif effect == "duotone":
        c1=spec.get("color1",(10,40,200)); c2=spec.get("color2",(240,220,60))
        if isinstance(c1,str): c1,c2=_parse_color(c1),_parse_color(c2)
        g=cv2.cvtColor(arr,cv2.COLOR_RGB2GRAY).astype(np.float32)/255
        r=np.zeros_like(arr,dtype=np.float32)
        for c in range(3): r[:,:,c]=tuple(c1)[c]*(1-g)+tuple(c2)[c]*g
        arr=r.clip(0,255).astype(np.uint8)
    elif effect == "gradient_map":
        s=spec.get("stops",[(0,0,0),(255,40,80),(255,200,50),(255,255,255)])
        g=cv2.cvtColor(arr,cv2.COLOR_RGB2GRAY).astype(np.float32)/255
        r=np.zeros((*g.shape,3),dtype=np.uint8)
        for c in range(3): r[:,:,c]=np.interp(g,[i/(len(s)-1) for i in range(len(s))],[sc[c] for sc in s]).clip(0,255).astype(np.uint8)
        arr=r
    elif effect == "color_boost":
        b=float(spec.get("boost",2)); t=spec.get("color","all")
        h=cv2.cvtColor(arr,cv2.COLOR_RGB2HSV).astype(np.float32)
        if t=="all": h[:,:,1]=np.clip(h[:,:,1]*b,0,255)
        elif t=="red": h[:,:,1][(h[:,:,0]<10)|(h[:,:,0]>170)]=np.clip(h[:,:,1][(h[:,:,0]<10)|(h[:,:,0]>170)]*b,0,255)
        elif t=="green": h[:,:,1][(h[:,:,0]>35)&(h[:,:,0]<85)]=np.clip(h[:,:,1][(h[:,:,0]>35)&(h[:,:,0]<85)]*b,0,255)
        elif t=="blue": h[:,:,1][(h[:,:,0]>100)&(h[:,:,0]<130)]=np.clip(h[:,:,1][(h[:,:,0]>100)&(h[:,:,0]<130)]*b,0,255)
        arr=cv2.cvtColor(h.astype(np.uint8),cv2.COLOR_HSV2RGB)
    elif effect == "channel_mix":
        rr,rg,rb=float(spec.get("rr",1)),float(spec.get("rg",0)),float(spec.get("rb",0))
        gr,gg,gb=float(spec.get("gr",0)),float(spec.get("gg",1)),float(spec.get("gb",0))
        br,bg,bb=float(spec.get("br",0)),float(spec.get("bg",0)),float(spec.get("bb",1))
        rc,gc,bc=arr.astype(np.float32).transpose(2,0,1)
        arr=np.stack([np.clip(rr*rc+rg*gc+rb*bc,0,255),np.clip(gr*rc+gg*gc+gb*bc,0,255),np.clip(br*rc+bg*gc+bb*bc,0,255)],axis=-1).astype(np.uint8)
    elif effect == "cross_process":
        s=float(spec.get("strength",0.5)); r,g,b=arr.astype(np.float32).transpose(2,0,1)
        c=np.stack([np.clip(r*1.1+g*0-b*0.1,0,255),np.clip(r*0+g*0.8+b*0.1,0,255),np.clip(r*0.1+g*0+b*1.2,0,255)],axis=-1).astype(np.uint8)
        arr=cv2.addWeighted(arr,1-s,c,s,0)
    elif effect == "sepia":
        s=float(spec.get("strength",1)); r,g,b=arr.astype(np.float32).transpose(2,0,1)
        sp=np.stack([np.clip(r*0.393+g*0.769+b*0.189,0,255),np.clip(r*0.349+g*0.686+b*0.168,0,255),np.clip(r*0.272+g*0.534+b*0.131,0,255)],axis=-1).astype(np.uint8)
        arr=cv2.addWeighted(arr,1-s,sp,s,0)
    elif effect == "temperature":
        t=float(spec.get("temperature",5000)); r,g,b=arr.astype(np.float32).transpose(2,0,1)
        w=(6500-t)/5500
        arr=np.stack([np.clip(r*(1+w*0.3),0,255),g,np.clip(b*(1-w*0.4),0,255)],axis=-1).astype(np.uint8)
    elif effect == "solarize":
        th=int(spec.get("threshold",128)); s=arr.copy(); s[arr>th]=255-s[arr>th]
        arr=cv2.addWeighted(arr,1-float(spec.get("strength",1)),s,float(spec.get("strength",1)),0)
    elif effect == "equalize":
        eq=cv2.equalizeHist(cv2.cvtColor(arr,cv2.COLOR_RGB2GRAY))
        l,a,b=cv2.split(cv2.cvtColor(arr,cv2.COLOR_RGB2LAB))
        arr=cv2.cvtColor(cv2.merge([eq,a,b]),cv2.COLOR_LAB2RGB)
    elif effect == "auto_contrast":
        l=float(spec.get("low",1)); h2=float(spec.get("high",99))
        af=arr.astype(np.float32)
        for c in range(3):
            ch=af[:,:,c]; lo,hi=np.percentile(ch,l),np.percentile(ch,h2)
            if hi>lo: ch=(ch-lo)/(hi-lo)*255
        arr=af.clip(0,255).astype(np.uint8)
    elif effect == "color_balance":
        l=float(spec.get("low",0.5)); h2=float(spec.get("high",99.5))
        af=arr.astype(np.float32)
        for c in range(3):
            ch=af[:,:,c]; lo,hi=np.percentile(ch,l),np.percentile(ch,h2)
            if hi>lo: ch=(ch-lo)/(hi-lo)*255
        arr=af.clip(0,255).astype(np.uint8)
    elif effect == "split_tone":
        sh=spec.get("shadow",(40,40,120)); hl=spec.get("highlight",(240,200,60)); b=float(spec.get("balance",0.5))
        if isinstance(sh,str): sh,hl=_parse_color(sh),_parse_color(hl)
        g=cv2.cvtColor(arr,cv2.COLOR_RGB2GRAY).astype(np.float32)/255
        r=arr.astype(np.float32)
        for c in range(3): r[:,:,c]=r[:,:,c]*0.5+(1-g)*np.array(sh,dtype=np.float32)[c]*(1-b)+g*np.array(hl,dtype=np.float32)[c]*b
        arr=r.clip(0,255).astype(np.uint8)
    elif effect == "bleach_bypass":
        g=cv2.cvtColor(arr,cv2.COLOR_RGB2GRAY).astype(np.float32)/255
        b=(arr.astype(np.float32)+255*g[:,:,None])*0.5
        arr=cv2.addWeighted(arr,1-float(spec.get("strength",0.5)),b.clip(0,255).astype(np.uint8),float(spec.get("strength",0.5)),0)
    elif effect == "teal_orange":
        s=float(spec.get("strength",0.6)); f=float(spec.get("fade",0.3))
        lab=cv2.cvtColor(arr,cv2.COLOR_RGB2LAB).astype(np.float32); l,a,b=cv2.split(lab)
        a=a*(1-f)+(l/255*40-a/128*30)*s
        b=b*(1-f)+(l/255*(-50)+(1-l/255)*30)*s
        arr=cv2.cvtColor(cv2.merge([l,a.clip(-128,127),b.clip(-128,127)]).astype(np.uint8),cv2.COLOR_LAB2RGB)
    elif effect == "invert":
        arr=255-arr
    elif effect == "threshold":
        th=int(spec.get("threshold",128)); mx=int(spec.get("max",255))
        mm={"binary":cv2.THRESH_BINARY,"binary_inv":cv2.THRESH_BINARY_INV,"trunc":cv2.THRESH_TRUNC,"tozero":cv2.THRESH_TOZERO,"tozero_inv":cv2.THRESH_TOZERO_INV}
        _,t=cv2.threshold(cv2.cvtColor(arr,cv2.COLOR_RGB2GRAY),th,mx,mm.get(spec.get("mode","binary"),cv2.THRESH_BINARY))
        arr=cv2.cvtColor(t,cv2.COLOR_GRAY2RGB)
    elif effect == "desaturate":
        a=float(spec.get("amount",0.5)); g=cv2.cvtColor(cv2.cvtColor(arr,cv2.COLOR_RGB2GRAY),cv2.COLOR_GRAY2RGB)
        arr=cv2.addWeighted(arr,1-a,g,a,0)
    elif effect == "monochrome":
        c=spec.get("color",(50,50,200))
        if isinstance(c,str): c=_parse_color(c)
        g=cv2.cvtColor(arr,cv2.COLOR_RGB2GRAY).astype(np.float32)/255
        arr=(g[:,:,None]*np.array(c,dtype=np.float32)[None,None,:]).clip(0,255).astype(np.uint8)
    elif effect == "glow":
        b=int(spec.get("blur",31)); a=float(spec.get("alpha",0.5)); th=int(spec.get("threshold",180))
        b+=1 if b%2==0 else 0
        br=(cv2.cvtColor(arr,cv2.COLOR_RGB2GRAY)>th).astype(np.float32)
        br=cv2.GaussianBlur(br,(b,b),0)
        bl=cv2.GaussianBlur(arr.astype(np.float32),(0,0),sigmaX=b//2)
        arr=cv2.addWeighted(arr,1,(bl*br[:,:,None]*a).clip(0,255).astype(np.uint8),1,0)
    elif effect == "rust":
        i=float(spec.get("intensity",0.3)); sc=int(spec.get("scale",50))
        n=cv2.resize(np.random.rand(h//sc+2,w//sc+2).astype(np.float32),(w,h),interpolation=cv2.INTER_LINEAR)
        m=(n>0.6).astype(np.float32)*i
        arr=(arr.astype(np.float32)*(1-m[:,:,None])+np.array([120,60,20],dtype=np.float32)*m[:,:,None]).clip(0,255).astype(np.uint8)
    elif effect == "canvas":
        s=float(spec.get("strength",0.15))
        t=np.ones((h,w),dtype=np.float32)
        for y in range(0,h,4): t[y:min(y+2,h)]=(0.95+np.sin(np.arange(w)*0.3)*0.05).astype(np.float32)
        for x in range(0,w,4): t[:,x:min(x+2,w)]*=(0.95+np.sin(np.arange(h)*0.3)*0.05)[:,None]
        t=t/t.max()
        arr=(arr.astype(np.float32)*(t*s+(1-s))).clip(0,255).astype(np.uint8)
    elif effect == "noise":
        a=float(spec.get("amount",0.1)); t=spec.get("type","gaussian")
        if t=="gaussian": n=np.random.randn(h,w,3).astype(np.float32)*a*50
        elif t=="uniform": n=(np.random.rand(h,w,3)-0.5).astype(np.float32)*a*255
        elif t=="speckle": n=arr.astype(np.float32)*np.random.randn(h,w,3).astype(np.float32)*a
        elif t=="salt_pepper":
            n=np.zeros((h,w,3),dtype=np.float32); s=np.random.rand(h,w)<a*0.5; p=np.random.rand(h,w)<a*0.5
            for c in range(3): n[:,:,c][s]=255-arr[s][:,c].astype(np.float32); n[:,:,c][p]=-arr[p][:,c].astype(np.float32)
        else: n=np.repeat(np.random.randn(h,w,1).astype(np.float32)*a*40,3,axis=2)
        arr=(arr.astype(np.float32)+n).clip(0,255).astype(np.uint8)
    elif effect == "frosted":
        k=int(spec.get("blur",15)); a=float(spec.get("noise",0.3))
        k+=1 if k%2==0 else 0
        n=(arr.astype(np.float32)+(np.random.rand(h,w,3)-0.5)*a*255).clip(0,255).astype(np.uint8)
        arr=cv2.GaussianBlur(n,(k,k),0)
    elif effect == "fisheye":
        s=float(spec.get("strength",1.0))
        yy,xx=np.mgrid[:h,:w].astype(np.float32); dx,dy=(xx-w/2)/(w/2),(yy-h/2)/(h/2)
        r2=dx**2+dy**2; sc=1+s*r2
        arr=cv2.remap(arr,np.clip(w/2+dx/sc*(w/2),0,w-1).astype(np.float32),np.clip(h/2+dy/sc*(h/2),0,h-1).astype(np.float32),cv2.INTER_LINEAR)
    elif effect == "cylinder":
        r=float(spec.get("radius",max(w,h)*0.8)); yy,xx=np.mgrid[:h,:w].astype(np.float32)
        t=(xx-w/2)/r
        arr=cv2.remap(arr,np.clip(w/2+r*np.sin(t),0,w-1).astype(np.float32),np.clip(yy*float(spec.get("stretch",1)),0,h-1).astype(np.float32),cv2.INTER_LINEAR)
    elif effect == "flag":
        a=float(spec.get("amplitude",20)); f=float(spec.get("frequency",0.03))
        yy,xx=np.mgrid[:h,:w].astype(np.float32)
        ax=a*(xx/w); off=ax*np.sin(yy*f*2+xx*f)
        arr=cv2.remap(arr,np.clip(xx+off,0,w-1).astype(np.float32),np.clip(yy+off*0.3,0,h-1).astype(np.float32),cv2.INTER_LINEAR)
    elif effect == "lens":
        b=float(spec.get("barrel",-0.2)); a=float(spec.get("astigmatism",0.1)); c=float(spec.get("chroma",2))
        yy,xx=np.mgrid[:h,:w].astype(np.float32); nx,ny=(xx-w/2)/(w/2),(yy-h/2)/(h/2); r=np.sqrt(nx**2+ny**2)
        bs=1+b*r**2; ax=1+a*ny**2; ay=1+a*nx**2
        arr=cv2.remap(arr,np.clip(w/2+nx*bs*ax*(w/2)+c*ny,0,w-1).astype(np.float32),np.clip(h/2+ny*bs*ay*(h/2),0,h-1).astype(np.float32),cv2.INTER_LINEAR)
    elif effect == "squeeze":
        ax=spec.get("axis","x"); a=float(spec.get("amount",0.4))
        yy,xx=np.mgrid[:h,:w].astype(np.float32)
        if ax=="x": arr=cv2.remap(arr,np.clip(w/2+(xx-w/2)/(1-a*np.sin(np.pi*yy/h)),0,w-1).astype(np.float32),yy.astype(np.float32),cv2.INTER_LINEAR)
        else: arr=cv2.remap(arr,xx.astype(np.float32),np.clip(h/2+(yy-h/2)/(1-a*np.sin(np.pi*xx/w)),0,h-1).astype(np.float32),cv2.INTER_LINEAR)
    elif effect == "tunnel":
        r=float(spec.get("radius",min(w,h)*0.4)); d=float(spec.get("depth",3))
        yy,xx=np.mgrid[:h,:w].astype(np.float32); dx,dy=xx-w/2,yy-h/2; dist=np.sqrt(dx**2+dy**2)
        sc=1/(1+(dist/r)**d)
        arr=cv2.remap(arr,np.clip(w/2+dx*sc,0,w-1).astype(np.float32),np.clip(h/2+dy*sc,0,h-1).astype(np.float32),cv2.INTER_LINEAR)
    elif effect == "polar":
        cx,cy=w/2,h/2; mr=min(cx,cy)
        yy,xx=np.mgrid[:h,:w].astype(np.float32); r=np.sqrt((xx-cx)**2+(yy-cy)**2); t=np.arctan2(yy-cy,xx-cx)
        arr=cv2.remap(arr,np.clip((t/math.pi+1)*0.5*w,0,w-1).astype(np.float32),np.clip(r/mr*h,0,h-1).astype(np.float32),cv2.INTER_LINEAR)
    elif effect == "pointillism":
        cs=int(spec.get("cell_size",6)); ds=float(spec.get("dot_size",0.6))
        r=np.ones((h,w,3),dtype=np.uint8)*255
        for y in range(0,h,cs):
            for x in range(0,w,cs):
                cy=min(y+cs//2,h-1); cx=min(x+cs//2,w-1)
                cv2.circle(r,(cx,cy),max(1,int(cs//2*ds)),tuple(int(c) for c in arr[cy,cx]),-1)
        arr=r
    elif effect == "crosshatch":
        sp=int(spec.get("spacing",12)); th=int(spec.get("thickness",1)); a1=float(spec.get("angle1",45)); a2=float(spec.get("angle2",135))
        g=cv2.cvtColor(arr,cv2.COLOR_RGB2GRAY); r=np.ones((h,w,3),dtype=np.uint8)*255
        for a in [a1,a2]:
            rad=math.radians(a); ca,sa=math.cos(rad),math.sin(rad)
            for i in range(-int(np.sqrt(w**2+h**2)),int(np.sqrt(w**2+h**2)),sp):
                x0,y0=int(i*ca),int(i*sa)
                pts=[(int(x0+t*ca),int(y0+t*sa)) for t in range(0,max(w,h),2) if 0<=int(x0+t*ca)<w and 0<=int(y0+t*sa)<h]
                if not pts: continue
                avg=sum(g[sy,sx] for sx,sy in pts)/len(pts)
                ld=1-avg/255
                if ld>0.1:
                    o=r.copy(); cv2.line(o,(int(x0-w*sa),int(y0+w*ca)),(int(x0+w*sa),int(y0-w*ca)),(0,0,0),th)
                    r=cv2.addWeighted(r,1-ld*0.6,o,ld*0.6,0)
        arr=r
    elif effect == "charcoal":
        b=int(spec.get("blur",15)); b+=1 if b%2==0 else 0
        g=cv2.cvtColor(arr,cv2.COLOR_RGB2GRAY)
        s=(cv2.GaussianBlur(g,(b,b),0).astype(np.float32)*0.6).clip(0,255).astype(np.uint8)
        e=cv2.Canny(g,40,120)
        arr=cv2.addWeighted(cv2.cvtColor(s,cv2.COLOR_GRAY2RGB),1,cv2.cvtColor(e,cv2.COLOR_GRAY2RGB),float(spec.get("intensity",0.3)),0)
    elif effect == "relief":
        s=float(spec.get("strength",1)); g=cv2.cvtColor(arr,cv2.COLOR_RGB2GRAY).astype(np.float32)
        r=(cv2.Sobel(g,cv2.CV_32F,1,0,ksize=3)+cv2.Sobel(g,cv2.CV_32F,0,1,ksize=3))*s+128
        arr=cv2.cvtColor(r.clip(0,255).astype(np.uint8),cv2.COLOR_GRAY2RGB)

    elif effect == "woodcut":
        b=int(spec.get("blur",7)); b+=1 if b%2==0 else 0; l=int(spec.get("levels",4)); st=255//(l-1)
        g=cv2.cvtColor(arr,cv2.COLOR_RGB2GRAY)
        p=(g.astype(np.float32)//st*st).clip(0,255).astype(np.uint8)
        e=cv2.dilate(cv2.Canny(cv2.GaussianBlur(g,(b,b),0),20,60),np.ones((2,2),dtype=np.uint8),iterations=1)
        r=cv2.cvtColor(p,cv2.COLOR_GRAY2RGB); r[e>0]=(0,0,0)
        if spec.get("color",True):
            for lv in range(l):
                lo=lv*st; hi=(lv+1)*st; m=(g>=lo)&(g<hi)
                if m.any(): r[m]=arr[m].mean(axis=0).astype(np.uint8)
        arr=r
    elif effect == "ink_wash":
        b=int(spec.get("blur",21)); b+=1 if b%2==0 else 0
        g=cv2.cvtColor(arr,cv2.COLOR_RGB2GRAY).astype(np.float32)
        w=cv2.GaussianBlur(g,(b,b),0); lo,hi=w.min(),w.max()
        if hi>lo: w=(w-lo)/(hi-lo)*255
        e=cv2.dilate(cv2.Canny(arr,20,60),np.ones((2,2),dtype=np.uint8),iterations=1)
        wd=(255-w*0.7).clip(0,255).astype(np.uint8)
        r=cv2.cvtColor(wd,cv2.COLOR_GRAY2RGB); r[e>0]=(20,20,30); arr=r
    elif effect == "scratchboard":
        g=cv2.cvtColor(arr,cv2.COLOR_RGB2GRAY).astype(np.float32)
        r=np.zeros((h,w,3),dtype=np.uint8)
        for _ in range(int(float(spec.get("density",500)))):
            x=random.randint(0,w-1); y=random.randint(0,h-1)
            if random.random()<g[y,x]/255:
                a=random.uniform(0,math.pi); l=random.randint(3,15)
                cv2.line(r,(x,y),(min(w-1,x+int(l*math.cos(a))),min(h-1,y+int(l*math.sin(a)))),(255,255,255),1)
        arr=r
    elif effect == "sobel":
        g=cv2.cvtColor(arr,cv2.COLOR_RGB2GRAY)
        m=np.sqrt(cv2.Sobel(g,cv2.CV_32F,1,0,ksize=int(spec.get("ksize",3)))**2+cv2.Sobel(g,cv2.CV_32F,0,1,ksize=int(spec.get("ksize",3)))**2)
        m=(m/m.max()*255).clip(0,255).astype(np.uint8)
        if spec.get("invert",False): m=255-m
        arr=cv2.cvtColor(m,cv2.COLOR_GRAY2RGB)
    elif effect == "laplacian":
        g=cv2.cvtColor(arr,cv2.COLOR_RGB2GRAY)
        l=np.abs(cv2.Laplacian(g,cv2.CV_32F,ksize=int(spec.get("ksize",3))))
        l=(l/l.max()*255).clip(0,255).astype(np.uint8)
        if spec.get("overlay",False): arr=cv2.addWeighted(arr,1-float(spec.get("alpha",0.3)),cv2.cvtColor(l,cv2.COLOR_GRAY2RGB),float(spec.get("alpha",0.3)),0)
        else: arr=cv2.cvtColor(l,cv2.COLOR_GRAY2RGB)
    elif effect == "ridge":
        g=cv2.cvtColor(arr,cv2.COLOR_RGB2GRAY).astype(np.float32)
        gx=cv2.Sobel(g,cv2.CV_32F,1,0,ksize=3); gy=cv2.Sobel(g,cv2.CV_32F,0,1,ksize=3)
        gxx=cv2.Sobel(gx,cv2.CV_32F,1,0,ksize=3); gyy=cv2.Sobel(gy,cv2.CV_32F,0,1,ksize=3)
        gxy=cv2.Sobel(gx,cv2.CV_32F,0,1,ksize=3)
        r=np.abs((gxx+gyy)**2-4*(gxx*gyy-gxy**2))
        r=(r/(r.max()+1e-8)*255).clip(0,255).astype(np.uint8)
        arr=cv2.cvtColor(r,cv2.COLOR_GRAY2RGB)
    elif effect == "unsharp":
        s=float(spec.get("sigma",3)); a=float(spec.get("amount",1.5))
        arr=cv2.addWeighted(arr,1+a,cv2.GaussianBlur(arr,(0,0),sigmaX=s),-a,0)
    elif effect == "bokeh":
        k=int(spec.get("ksize",31)); k+=1 if k%2==0 else 0
        kn=np.zeros((k,k),dtype=np.float32)
        for y2 in range(k):
            for x2 in range(k):
                if (x2-k//2)**2+(y2-k//2)**2<=(k//2)**2: kn[y2,x2]=1
        kn=kn/kn.sum(); r,g,b=cv2.split(arr.astype(np.float32))
        arr=cv2.merge([cv2.filter2D(r,-1,kn),cv2.filter2D(g,-1,kn),cv2.filter2D(b,-1,kn)]).clip(0,255).astype(np.uint8)
    elif effect == "lens_flare":
        cx,cy=w//2,h//2; i=float(spec.get("intensity",0.4)); ng=int(spec.get("ghosts",5))
        r=arr.astype(np.float32)
        for g in range(ng):
            t=(g+1)/(ng+1); gx=int(cx+(w/2-cx)*t*1.5); gy=int(cy+(h/2-cy)*t*1.5); rad=int(30+t*80)
            o=np.zeros_like(r); cv2.circle(o,(gx,gy),rad,(255,220,150),-1)
            r+=o*i*(1-t)*0.3
        cv2.circle(r,(cx,cy),20,(255,200,150),-1)
        arr=r.clip(0,255).astype(np.uint8)
    elif effect == "light_leak":
        c=spec.get("color",(220,100,50))
        if isinstance(c,str): c=_parse_color(c)
        i=float(spec.get("intensity",0.3)); s=float(spec.get("strength",1)); random.seed(int(spec.get("seed",99)))
        x1,y1=random.randint(0,w),random.randint(0,h//3); x2,y2=random.randint(0,w),random.randint(h//2,h)
        o=np.zeros((h,w,3),dtype=np.float32)
        cv2.line(o,(x1,y1),(x2,y2),c,max(10,int(80*s)))
        o=cv2.GaussianBlur(o,(0,0),sigmaX=30)
        arr=cv2.addWeighted(arr,1,o.clip(0,255).astype(np.uint8),i,0)
    elif effect == "rain":
        d=float(spec.get("density",100)); l=float(spec.get("length",20)); a=float(spec.get("angle",0.2)); random.seed(42)
        o=np.zeros((h,w,3),dtype=np.uint8)
        dx=int(l*math.sin(a)); dy=int(l*math.cos(a))
        for _ in range(int(d*w/100)):
            x=random.randint(0,w-1); y=random.randint(0,h-1)
            cv2.line(o,(x,y),(min(w-1,max(0,x+dx)),min(h-1,y+dy)),(200,200,220),1)
        if float(spec.get("blur",0))>0: o=cv2.GaussianBlur(o.astype(np.float32),(int(float(spec.get("blur",0)))|1,)*2,0).clip(0,255).astype(np.uint8)
        arr=cv2.addWeighted(arr,1,o,float(spec.get("alpha",0.4)),0)
    elif effect == "fog":
        d=float(spec.get("density",0.3)); c=spec.get("color",(200,200,220))
        if isinstance(c,str): c=_parse_color(c)
        n=cv2.resize(np.random.rand(max(2,h//20+2),max(2,w//20+2)).astype(np.float32),(w,h),interpolation=cv2.INTER_LINEAR)
        v=d*(0.5+0.5*n); fc=np.array(c,dtype=np.float32)
        arr=(arr.astype(np.float32)*(1-v[:,:,None])+fc[None,None,:]*v[:,:,None]).clip(0,255).astype(np.uint8)
    elif effect == "heat_haze":
        a=float(spec.get("amplitude",5)); f=float(spec.get("frequency",0.08))
        yy,xx=np.mgrid[:h,:w].astype(np.float32)
        off=a*(yy/h)*np.sin(xx*f+yy*0.02)
        arr=cv2.remap(arr,np.clip(xx+off,0,w-1).astype(np.float32),yy.astype(np.float32),cv2.INTER_LINEAR)
    elif effect == "vortex":
        a=float(spec.get("angle",math.pi)); r=float(spec.get("radius",min(w,h)*0.5)); cx,cy=w/2,h/2
        yy,xx=np.mgrid[:h,:w].astype(np.float32); dx,dy=xx-cx,yy-cy; d=np.sqrt(dx**2+dy**2)
        t=a*np.exp(-d/r)
        arr=cv2.remap(arr,np.clip(cx+dx*np.cos(t)-dy*np.sin(t),0,w-1).astype(np.float32),np.clip(cy+dx*np.sin(t)+dy*np.cos(t),0,h-1).astype(np.float32),cv2.INTER_LINEAR)
    elif effect == "crumple":
        a=float(spec.get("amplitude",8)); random.seed(int(spec.get("seed",42)))
        yy,xx=np.mgrid[:h,:w].astype(np.float32)
        dx=cv2.GaussianBlur(np.random.randn(h,w).astype(np.float32),(0,0),sigmaX=20)*a
        dy=cv2.GaussianBlur(np.random.randn(h,w).astype(np.float32),(0,0),sigmaX=20)*a
        arr=cv2.remap(arr,np.clip(xx+dx,0,w-1).astype(np.float32),np.clip(yy+dy,0,h-1).astype(np.float32),cv2.INTER_LINEAR)
    elif effect == "dream":
        b=int(spec.get("blur",21)); b+=1 if b%2==0 else 0; s=float(spec.get("strength",0.5))
        bl=cv2.GaussianBlur(arr,(b,b),0)
        hsv=cv2.cvtColor(bl,cv2.COLOR_RGB2HSV).astype(np.float32); hsv[:,:,1]=hsv[:,:,1]*1.3
        bl=cv2.cvtColor(hsv.clip(0,255).astype(np.uint8),cv2.COLOR_HSV2RGB)
        arr=cv2.addWeighted(arr,1-s,bl,s,30)
    elif effect == "film_stock":
        g=float(spec.get("grain",0.15)); c=float(spec.get("contrast",1.2)); s=float(spec.get("saturation",0.8))
        af=arr.astype(np.float32)/255; af=(af-0.5)*c+0.5
        hsv=cv2.cvtColor((af*255).clip(0,255).astype(np.uint8),cv2.COLOR_RGB2HSV).astype(np.float32)
        hsv[:,:,1]=hsv[:,:,1]*s
        af=cv2.cvtColor(hsv.clip(0,255).astype(np.uint8),cv2.COLOR_HSV2RGB).astype(np.float32)/255
        af+=np.random.randn(h,w,3).astype(np.float32)*g*0.3; af[:,:,2]*=0.95
        arr=(af.clip(0,1)*255).astype(np.uint8)
    elif effect == "oil_painting":
        r=int(spec.get("radius",5)); l=int(spec.get("levels",10))
        try: arr=cv2.xphoto.oilPainting(arr,r,l)
        except: arr=cv2.bilateralFilter(cv2.medianBlur(arr,min(r*2+1,49)),min(r*2+1,49),80,80)
    elif effect == "edge_glow":
        b=int(spec.get("blur",11)); b+=1 if b%2==0 else 0
        e=cv2.dilate(cv2.Canny(cv2.cvtColor(arr,cv2.COLOR_RGB2GRAY),int(spec.get("low",20)),int(spec.get("high",80))),np.ones((2,2),dtype=np.uint8),iterations=1)
        g=cv2.GaussianBlur(e.astype(np.float32),(b,b),0)/255
        c=spec.get("color",(80,200,255))
        if isinstance(c,str): c=_parse_color(c)
        r=arr.astype(np.float32)
        for ch in range(3): r[:,:,ch]+=g*c[ch]*float(spec.get("intensity",0.6))
        arr=r.clip(0,255).astype(np.uint8)
    elif effect == "infrared":
        s=float(spec.get("strength",0.8)); r,g,b=cv2.split(arr.astype(np.float32))
        ir=np.stack([b*0.6+g*0.4,g*0.8+r*0.2,r*0.5+g*0.5],axis=-1)
        ir=cv2.addWeighted(ir,0.7,cv2.GaussianBlur(ir,(0,0),sigmaX=5),0.3,0)
        arr=cv2.addWeighted(arr,1-s,ir.clip(0,255).astype(np.uint8),s,0)
    elif effect == "xray":
        s=float(spec.get("strength",1)); g=cv2.cvtColor(arr,cv2.COLOR_RGB2GRAY).astype(np.float32)
        inv=255-g; lo,hi=np.percentile(inv,2),np.percentile(inv,98)
        if hi>lo: inv=(inv-lo)/(hi-lo)*255
        r=np.stack([inv.clip(0,255).astype(np.uint8),(inv*0.85).clip(0,255).astype(np.uint8),(inv*0.6).clip(0,255).astype(np.uint8)],axis=-1)
        arr=cv2.addWeighted(arr,1-s,r,s,0)
    elif effect == "tilt_shift":
        sw=float(spec.get("sharp_width",0.3)); b=int(spec.get("blur",21)); b+=1 if b%2==0 else 0
        gr=np.abs(np.linspace(-1,1,h)); ba=np.clip((np.abs(gr)-sw)/(1-sw),0,1)
        r=arr.copy().astype(np.float32)
        for row in range(h):
            if ba[row]>0:
                ks=max(3,int(b*ba[row])|1)
                r[row:row+1]=cv2.addWeighted(arr[row:row+1].astype(np.float32),1-ba[row],cv2.GaussianBlur(arr[row:row+1],(ks,ks),0).astype(np.float32),ba[row],0)
        arr=r.clip(0,255).astype(np.uint8)
    elif effect == "motion_blur":
        a=float(spec.get("angle",0)); d=int(spec.get("distance",20))
        r=math.radians(a); dx,dy=max(1,int(abs(d*math.cos(r)))),max(1,int(abs(d*math.sin(r))))
        kn=np.zeros((dy*2+1,dx*2+1),dtype=np.float32)
        for t in range(-d,d+1):
            xt,yt=int(t*math.cos(r))+dx,int(t*math.sin(r))+dy
            if 0<=xt<kn.shape[1] and 0<=yt<kn.shape[0]: kn[yt,xt]=1
        arr=cv2.filter2D(arr,-1,kn/kn.sum())
    elif effect == "zoom_blur":
        a=float(spec.get("amount",0.3)); steps=max(3,int(a*20)); r=np.zeros_like(arr,dtype=np.float32)
        for s in range(steps):
            sc=1-(s/steps)*a
            if sc<=0.01: break
            nw,nh=int(w*sc),int(h*sc)
            if nw<2 or nh<2: break
            sm=cv2.resize(arr,(nw,nh),interpolation=cv2.INTER_LINEAR)
            xo,yo=(w-nw)//2,(h-nh)//2
            r[yo:yo+nh,xo:xo+nw]+=sm.astype(np.float32)
        arr=(r/steps).clip(0,255).astype(np.uint8)
    elif effect == "pinhole":
        yy,xx=np.mgrid[:h,:w]; d=np.sqrt((xx-w/2)**2+(yy-h/2)**2); md=np.sqrt((w/2)**2+(h/2)**2)
        v=(1-(d/md)*0.7).clip(0.3,1)
        a=cv2.GaussianBlur(arr,(5,5),0).astype(np.float32)
        a[:,:,0]*=0.9; a[:,:,2]*=1.1
        for c in range(3): a[:,:,c]*=v
        arr=a.clip(0,255).astype(np.uint8)
    elif effect == "polaroid":
        s=min(w,h); xo,yo=(w-s)//2,(h-s)//2
        sq=arr[yo:yo+s,xo:xo+s]
        hsv=cv2.cvtColor(sq,cv2.COLOR_RGB2HSV).astype(np.float32)
        hsv[:,:,0]=(hsv[:,:,0]+5)%180; hsv[:,:,1]*=0.85
        sq=cv2.cvtColor(hsv.clip(0,255).astype(np.uint8),cv2.COLOR_HSV2RGB)
        b=int(s*0.05); r=np.ones((s+b*2,s+b*2,3),dtype=np.uint8)*245
        r[b:b+s,b:b+s]=sq; arr=cv2.resize(r,(w,h),interpolation=cv2.INTER_LINEAR)
    elif effect == "vintage":
        s=float(spec.get("strength",0.6))
        a=arr.astype(np.float32); g=cv2.cvtColor(arr,cv2.COLOR_RGB2GRAY).astype(np.float32)
        a=a*0.7+g[:,:,None]*0.3; a[:,:,0]*=0.85; a[:,:,2]*=1.15; a=a*0.9+25
        arr=cv2.addWeighted(arr,1-s,a.clip(0,255).astype(np.uint8),s,0)
    elif effect == "lomo":
        hsv=cv2.cvtColor(arr,cv2.COLOR_RGB2HSV).astype(np.float32)
        hsv[:,:,1]*=1.4; hsv[:,:,2]*=1.1
        a=cv2.cvtColor(hsv.clip(0,255).astype(np.uint8),cv2.COLOR_HSV2RGB).astype(np.float32)
        yy,xx=np.mgrid[:h,:w]; d=np.sqrt((xx-w/2)**2+(yy-h/2)**2); md=np.sqrt((w/2)**2+(h/2)**2)
        v=np.clip(1-(d/md)**2*0.8,0,1); a*=v[:,:,None]
        a[:,:,0]*=0.95; a[:,:,1]*=1.05
        arr=cv2.addWeighted(arr,1-float(spec.get("strength",1)),a.clip(0,255).astype(np.uint8),float(spec.get("strength",1)),0)
    elif effect == "hdr":
        s=float(spec.get("strength",0.7))
        d=cv2.detailEnhance(arr,sigma_s=10,sigma_r=0.15)
        lab=cv2.cvtColor(d,cv2.COLOR_RGB2LAB).astype(np.float32); l,a,b=cv2.split(lab)
        l=cv2.createCLAHE(clipLimit=3,tileGridSize=(8,8)).apply(l.astype(np.uint8))
        hdr=cv2.cvtColor(cv2.merge([l.astype(np.float32),a,b]).astype(np.uint8),cv2.COLOR_LAB2RGB)
        hsv=cv2.cvtColor(hdr,cv2.COLOR_RGB2HSV).astype(np.float32); hsv[:,:,1]*=1.2
        hdr=cv2.cvtColor(hsv.clip(0,255).astype(np.uint8),cv2.COLOR_HSV2RGB)
        arr=cv2.addWeighted(arr,1-s,hdr,s,0)
    elif effect == "vhs":
        r,g,b=cv2.split(arr.astype(np.float32))
        b=cv2.GaussianBlur(b,(5,5),0); r=cv2.GaussianBlur(r,(3,3),0)
        a=cv2.merge([r,g,b]).clip(0,255).astype(np.uint8)
        a=cv2.addWeighted(a,0.7,np.roll(arr,4,axis=1),0.3,0)
        for i in range(0,h,2): a[i:i+1]=(a[i:i+1].astype(np.float32)*0.65).astype(np.uint8)
        a=cv2.addWeighted(a,0.95,(np.random.randn(h,w,3).astype(np.float32)*8).clip(0,255).astype(np.uint8),0.05,0)
        arr=a
    elif effect == "scanlines":
        th=int(spec.get("thickness",1)); sp=int(spec.get("spacing",3)); op=float(spec.get("opacity",0.25))
        r=arr.copy().astype(np.float32)
        for i in range(0,h,sp): r[i:min(i+th,h)]*=1-op
        if spec.get("interlaced",False):
            for i in range(0,h,2): r[i:i+1]=cv2.GaussianBlur(r[i:i+1],(5,5),0)
        arr=r.clip(0,255).astype(np.uint8)
    elif effect == "pixel_sort":
        i=float(spec.get("intensity",0.3)); g=cv2.cvtColor(arr,cv2.COLOR_RGB2GRAY); st=max(1,int((1-i)*20))
        for y in range(0,h,st):
            in_run=False; run_start=0
            for x in range(w):
                bright=g[y,x]>int(spec.get("threshold",100))
                if bright and not in_run: in_run=True; run_start=x
                elif (not bright or x==w-1) and in_run:
                    run_end=x if not bright else x+1
                    if run_end-run_start>2:
                        order=np.argsort(g[y,run_start:run_end])
                        arr[y,run_start:run_end]=arr[y,run_start:run_end][order]
                    in_run=False
    elif effect == "chromatic_ab":
        s=int(spec.get("shift",5)); d=spec.get("direction","h")
        r,g,b=cv2.split(arr)
        if d=="h": r=np.roll(r.astype(np.float32),s,1); b=np.roll(b.astype(np.float32),-s,1)
        elif d=="v": r=np.roll(r.astype(np.float32),s,0); b=np.roll(b.astype(np.float32),-s,0)
        else:
            yy,xx=np.mgrid[:h,:w].astype(np.float32)
            dist=np.sqrt((xx-w/2)**2+(yy-h/2)**2)/np.sqrt((w/2)**2+(h/2)**2)
            sm=(dist*s).astype(np.int32)
            for y in range(h):
                for x in range(w):
                    if sm[y,x]>0 and x+sm[y,x]<w: r[y,x]=arr[y,min(x+sm[y,x],w-1),0].astype(np.float32); b[y,x]=arr[y,max(x-sm[y,x],0),2].astype(np.float32)
        arr=cv2.merge([r.clip(0,255).astype(np.uint8),g,b.clip(0,255).astype(np.uint8)])
    elif effect == "dither":
        m=spec.get("method","floyd_steinberg"); l=int(spec.get("levels",4)); st=255//(l-1)
        g=cv2.cvtColor(arr,cv2.COLOR_RGB2GRAY).astype(np.float32)
        r=np.zeros_like(g)
        if m=="floyd_steinberg":
            for y in range(h):
                for x in range(w):
                    o=g[y,x]; n=round(o/st)*st; r[y,x]=n; e=o-n
                    if x+1<w: g[y,x+1]+=e*7/16
                    if y+1<h:
                        if x>0: g[y+1,x-1]+=e*3/16
                        g[y+1,x]+=e*5/16
                        if x+1<w: g[y+1,x+1]+=e/16
            arr=cv2.cvtColor(r.clip(0,255).astype(np.uint8),cv2.COLOR_GRAY2RGB)
        elif m=="atkinson":
            for y in range(h):
                for x in range(w):
                    o=g[y,x]; n=round(o/st)*st; r[y,x]=n; e=(o-n)//8
                    for dx,dy in [(1,0),(2,0),(-1,1),(0,1),(1,1),(0,2)]:
                        nx,ny=x+dx,y+dy
                        if 0<=nx<w and 0<=ny<h: g[ny,nx]+=e
            arr=cv2.cvtColor(r.clip(0,255).astype(np.uint8),cv2.COLOR_GRAY2RGB)
    elif effect == "anaglyph":
        s=int(spec.get("shift",10)); r,g,b=cv2.split(arr)
        r=np.roll(r.astype(np.float32),-s,1)
        rn=np.zeros_like(arr,dtype=np.float32); rn[:,:,0]=r; rn[:,:,1]=g.astype(np.float32)*0.5; rn[:,:,2]=b.astype(np.float32)*0.5
        arr=rn.clip(0,255).astype(np.uint8)
    elif effect == "comic_book":
        g=cv2.cvtColor(arr,cv2.COLOR_RGB2GRAY); l=int(spec.get("levels",6)); st=255//(l-1)
        p=(arr.astype(np.float32)//st*st+st//2).clip(0,255).astype(np.uint8)
        e=cv2.dilate(cv2.Canny(g,20,80),np.ones((2,2),dtype=np.uint8),iterations=1)
        p[e>0]=(0,0,0)
        if spec.get("halftone",False):
            gh=cv2.cvtColor(p,cv2.COLOR_RGB2GRAY)
            for y in range(0,h,6):
                for x in range(0,w,6):
                    if gh[y:y+6,x:x+6].mean()<200: cv2.circle(p,(x+3,y+3),max(1,int((1-gh[y:y+6,x:x+6].mean()/255)*3)),(0,0,0),-1)
        arr=p
    elif effect == "topographic":
        l=int(spec.get("levels",8)); th=int(spec.get("thickness",1)); st=255/l
        g=cv2.cvtColor(arr,cv2.COLOR_RGB2GRAY).astype(np.float32)
        r=arr.copy()
        for lv in range(1,l):
            val=lv*st; m=(g>=val-th*5)&(g<=val+th*5); r[m]=(0,0,0)
        for lv in range(l):
            lo=lv*st; hi=(lv+1)*st; m=(g>=lo)&(g<hi)
            if m.any():
                t=lv/l; r[m]=(int(34+t*180),int(139-t*50),int(34-t*20))  # BGR
        arr=r
    elif effect == "embroidery":
        st=int(spec.get("stitch",8)); g=cv2.cvtColor(arr,cv2.COLOR_RGB2GRAY)
        r=np.ones((h,w,3),dtype=np.uint8)*250
        for y in range(0,h,st):
            for x in range(0,w,st):
                y2=min(y+st-1,h-1); x2=min(x+st-1,w-1)
                cy=min(y+st//2,h-1); cx=min(x+st//2,w-1); c=tuple(int(ci) for ci in arr[cy,cx])
                if ((y//st)+(x//st))%2==0: cv2.line(r,(x,y),(x2,y2),c,max(1,st//4))
                else: cv2.line(r,(x2,y),(x,y2),c,max(1,st//4))
        arr=r
    elif effect == "edges_color":
        g=cv2.cvtColor(arr,cv2.COLOR_RGB2GRAY).astype(np.float32)
        def norm(a): return (np.abs(a)/(np.abs(a).max()+1e-8)*255).clip(0,255).astype(np.uint8)
        arr=cv2.merge([norm(cv2.Sobel(g,cv2.CV_32F,1,0,ksize=3)),norm(cv2.Sobel(g,cv2.CV_32F,0,1,ksize=3)),norm(cv2.Sobel(g,cv2.CV_32F,1,1,ksize=3))])
    elif effect == "prism":
        i=float(spec.get("intensity",0.5)); off=int(i*30); r,g,b=cv2.split(arr)
        r=np.roll(r.astype(np.float32),off,1); r=np.roll(r,off//2,0)
        b=np.roll(b.astype(np.float32),-off,1); b=np.roll(b,-off//2,0)
        p=cv2.merge([r.clip(0,255).astype(np.uint8),g,b.clip(0,255).astype(np.uint8)])
        arr=cv2.addWeighted(arr,1-float(spec.get("alpha",0.5)),p,float(spec.get("alpha",0.5)),0)
    elif effect == "map":
        s=spec.get("style","terrain"); g=cv2.cvtColor(arr,cv2.COLOR_RGB2GRAY).astype(np.float32)
        g=(g-g.min())/(g.max()-g.min()+1e-8); r=np.zeros((h,w,3),dtype=np.uint8)
        if s=="terrain":
            for y in range(h):
                for x in range(w):
                    v=g[y,x]
                    if v<0.2: r[y,x]=(180,160,80)
                    elif v<0.45: r[y,x]=(60,160,40)
                    elif v<0.65: r[y,x]=(40,120,100)
                    elif v<0.85: r[y,x]=(40,80,140)
                    else: r[y,x]=(200,200,220)
        elif s=="heat":
            for y in range(h):
                for x in range(w):
                    v=g[y,x]; r[y,x]=(int(255*(1-v)),min(int(255*(1-abs(v-0.5)*2)),200),int(255*v))
        elif s=="relief":
            sx=cv2.Sobel(g,cv2.CV_32F,1,0,ksize=3); sy=cv2.Sobel(g,cv2.CV_32F,0,1,ksize=3)
            r=cv2.cvtColor(((sx*0.5+sy*0.5+1)*128).clip(0,255).astype(np.uint8),cv2.COLOR_GRAY2RGB)
        arr=r
    elif effect == "smoke":
        d=float(spec.get("density",0.3)); random.seed(42)
        o=np.zeros((h,w,3),dtype=np.uint8)
        for _ in range(int(d*30)):
            x,y=random.randint(0,w),random.randint(0,h)
            cx,cy=x,y
            for _ in range(random.randint(50,200)):
                cx+=random.randint(-3,3); cy-=random.randint(0,2)
                cv2.circle(o,(np.clip(int(cx),0,w-1),np.clip(int(cy),0,h-1)),random.randint(3,8),(200,200,210),-1)
        o=cv2.GaussianBlur(o.astype(np.float32),(0,0),sigmaX=10)
        arr=cv2.addWeighted(arr.astype(np.float32),1,o.astype(np.float32),float(spec.get("alpha",0.4)),0).clip(0,255).astype(np.uint8)
    elif effect == "sparkle":
        d=int(spec.get("density",150)); random.seed(1)
        o=np.zeros((h,w,3),dtype=np.uint8)
        for _ in range(d):
            x,y=random.randint(0,w-1),random.randint(0,h-1)
            if random.random()<cv2.cvtColor(arr,cv2.COLOR_RGB2GRAY)[y,x]/255:
                sz=random.randint(1,3)
                for dx,dy in [(0,0),(1,0),(-1,0),(0,1),(0,-1)]:
                    o[np.clip(y+dy*sz,0,h-1),np.clip(x+dx*sz,0,w-1)]=(255,255,220)
        arr=cv2.addWeighted(arr,1,o,1,0)
    elif effect == "starburst":
        th=int(spec.get("threshold",180)); g=cv2.cvtColor(arr,cv2.COLOR_RGB2GRAY)
        b=(g>th).astype(np.uint8); b=cv2.dilate(b,np.ones((5,5),dtype=np.uint8),iterations=1)
        nl,_,_,ct=cv2.connectedComponentsWithStats(b,connectivity=8)
        r=arr.copy().astype(np.float32)
        for i in range(1,nl):
            cx,cy=int(ct[i][0]),int(ct[i][1])
            for a in range(0,360,45):
                ra=math.radians(a); l=40+g[cy,cx]/255*30
                for t in range(1,int(l)):
                    xt,yt=int(cx+t*math.cos(ra)),int(cy+t*math.sin(ra))
                    if 0<=xt<w and 0<=yt<h: r[yt,xt]+=np.array([255*(1-t/l),255*(1-t/l)*0.9,200*(1-t/l)])
        arr=r.clip(0,255).astype(np.uint8)
    elif effect == "water_drop":
        n=int(spec.get("num_drops",12)); random.seed(7)
        r=arr.copy().astype(np.float32)
        for _ in range(n):
            cx=random.randint(w//4,3*w//4); cy=random.randint(h//4,3*h//4); rad=random.randint(10,40)
            yy,xx=np.mgrid[:h,:w].astype(np.float32); dx,dy=xx-cx,yy-cy; d=np.sqrt(dx**2+dy**2)
            m=d<rad; src_x=np.clip(cx+dx/1.5,0,w-1).astype(np.float32); src_y=np.clip(cy+dy/1.5,0,h-1).astype(np.float32)
            w2=cv2.remap(arr,src_x,src_y,cv2.INTER_LINEAR).astype(np.float32)
            w2[(d>rad-2)&(d<rad+1)]+=40; r[m]=w2[m]
        arr=r.clip(0,255).astype(np.uint8)
    elif effect == "glass":
        a=float(spec.get("amplitude",3)); random.seed(42)
        yy,xx=np.mgrid[:h,:w].astype(np.float32)
        dx=cv2.resize(np.random.randn(max(2,h//40+2),max(2,w//40+2)).astype(np.float32),(w,h),interpolation=cv2.INTER_CUBIC)*a
        dy=cv2.resize(np.random.randn(max(2,h//40+2),max(2,w//40+2)).astype(np.float32),(w,h),interpolation=cv2.INTER_CUBIC)*a
        arr=cv2.remap(arr,np.clip(xx+dx,0,w-1).astype(np.float32),np.clip(yy+dy,0,h-1).astype(np.float32),cv2.INTER_LINEAR)
    elif effect == "circuit":
        d=float(spec.get("density",0.5)); g=cv2.cvtColor(arr,cv2.COLOR_RGB2GRAY).astype(np.float32)
        o=np.zeros((h,w,3),dtype=np.uint8); random.seed(17)
        for _ in range(int(d*40)):
            x1,y1=random.randint(0,w),random.randint(0,h)
            if random.random()>g[min(y1,h-1),min(x1,w-1)]/255: continue
            x2,y2=(random.randint(0,w),y1) if random.random()<0.5 else (x1,random.randint(0,h))
            c=(40,random.randint(120,200),40)
            cv2.line(o,(x1,y1),(x2,y2),c,1); cv2.circle(o,(x1,y1),2,(60,200,60),-1); cv2.circle(o,(x2,y2),2,(60,200,60),-1)
        arr=cv2.addWeighted(arr,1,o,float(spec.get("alpha",0.6)),0)

    elif effect == "cyanotype":
        s=float(spec.get("strength",1)); g=cv2.cvtColor(arr,cv2.COLOR_RGB2GRAY).astype(np.float32)/255
        cy=np.stack([(g*40+30).clip(0,255).astype(np.uint8),(g*60+80).clip(0,255).astype(np.uint8),(g*100+155).clip(0,255).astype(np.uint8)],axis=-1)
        arr=cv2.addWeighted(arr,1-s,cy,s,0)
    elif effect == "kodalith":
        c=float(spec.get("contrast",3)); t=spec.get("tint",None); th=float(spec.get("threshold",0.5))
        g=cv2.cvtColor(arr,cv2.COLOR_RGB2GRAY).astype(np.float32)/255; k=(np.clip((g-0.5)*c+0.5,0,1)>th).astype(np.float32)
        if t:
            if isinstance(t,str): t=_parse_color(t)
            k3=np.zeros((h,w,3),dtype=np.uint8); k3[k>0]=t; k3[k==0]=(245,245,245)
            arr=k3
        else: arr=cv2.cvtColor((k*255).clip(0,255).astype(np.uint8),cv2.COLOR_GRAY2RGB)
    elif effect == "emulsion_lift":
        s=float(spec.get("strength",0.7)); random.seed(42); a=arr.astype(np.float32)
        g=cv2.cvtColor(arr,cv2.COLOR_RGB2GRAY).astype(np.float32)/255; a=a*0.6+g[:,:,None]*128*0.4
        dx=cv2.GaussianBlur(np.random.randn(h,w).astype(np.float32),(0,0),sigmaX=25)*6
        dy=cv2.GaussianBlur(np.random.randn(h,w).astype(np.float32),(0,0),sigmaX=25)*6
        yy,xx=np.mgrid[:h,:w].astype(np.float32)
        a=cv2.remap(a,np.clip(xx+dx,0,w-1).astype(np.float32),np.clip(yy+dy,0,h-1).astype(np.float32),cv2.INTER_LINEAR)
        bm=np.ones((h,w),dtype=np.uint8)*255; cx,cy=w//2,h//2; rn=cv2.GaussianBlur(np.random.randn(h,w).astype(np.float32),(0,0),sigmaX=15)
        for y in range(h):
            for x in range(w):
                if np.sqrt(((x-cx)/cx)**2+((y-cy)/cy)**2)>0.85+rn[y,x]*0.1: bm[y,x]=0
        a[bm==0]=(240,240,235); a[:,:,0]*=0.92; a[:,:,2]*=1.08
        arr=cv2.addWeighted(arr,1-s,a.clip(0,255).astype(np.uint8),s,0)
    elif effect == "double_exposure":
        b=spec.get("blend","screen"); sx=int(spec.get("shift_x",30)); sy=int(spec.get("shift_y",20))
        sh=np.roll(arr,sy,0); sh=np.roll(sh,sx,1)
        if spec.get("ghost_blur",True): sh=cv2.GaussianBlur(sh.astype(np.float32),(9,9),0)
        sc=cv2.resize(cv2.resize(arr,(w//3,h//3)),(w,h),interpolation=cv2.INTER_LINEAR).astype(np.float32)
        gc=cv2.GaussianBlur((255-arr).astype(np.float32),(15,15),0)
        af,sf=arr.astype(np.float32),sc; gf=gc
        if b=="screen":
            r=255-(255-af)*(255-sf)/255
            cm=np.zeros((h,w),dtype=np.float32); cv2.circle(cm,(w//2,h//2),min(w,h)//4,1,-1)
            cm=cv2.GaussianBlur(cm,(31,31),0)
            r=r*(1-cm[:,:,None]*0.3)+gf*cm[:,:,None]*0.3
        elif b=="multiply": r=af*sf/255
        else: r=np.where(af<128,2*af*sf/255,255-2*(255-af)*(255-sf)/255)
        arr=cv2.addWeighted(arr,1-float(spec.get("alpha",1)),r.clip(0,255).astype(np.uint8),float(spec.get("alpha",1)),0)
    elif effect == "knit":
        sc=int(spec.get("scale",6))
        r=np.ones((h,w,3),dtype=np.uint8)*245
        for y in range(0,h,sc*2):
            for x in range(-sc//2,w+sc,sc):
                cy=min(y+sc//2,h-1); cx=np.clip(x+sc//2,0,w-1)
                c=tuple(int(ci)for ci in arr[cy,cx])
                if ((y//(sc*2))+(x//sc))%2==0: pts=[(x,y),(x+sc//2,y+sc),(x+sc,y)]
                else: pts=[(x,y+sc),(x+sc//2,y),(x+sc,y+sc)]
                cv2.line(r,pts[0],pts[1],c,max(1,sc//3)); cv2.line(r,pts[1],pts[2],c,max(1,sc//3))
        arr=r
    elif effect == "chainmail":
        r2=int(spec.get("radius",8)); t2=int(spec.get("thickness",2))
        r=np.zeros((h,w,3),dtype=np.uint8); sp=r2*2+t2
        for y in range(-r2,h+r2,sp):
            for x in range(-r2,w+r2,sp):
                ox=r2 if(y//sp)%2==0 else 0
                cy=np.clip(y+r2,0,h-1); cx=np.clip(x+ox,0,w-1)
                c=tuple(int(ci)for ci in arr[cy,min(cx,w-1)])
                cv2.circle(r,(min(x+ox,w-1),min(y+r2,h-1)),r2,c,t2)
        arr=r
    elif effect == "marbling":
        sc=float(spec.get("scale",0.02)); sw=float(spec.get("swirl",3)); a=float(spec.get("alpha",0.7))
        yy,xx=np.mgrid[:h,:w].astype(np.float32)
        p=np.sin(xx*sc+sw*np.sin(yy*sc*2))*np.cos(yy*sc+sw*np.sin(xx*sc*1.5))
        p=(p+1)/2; r=np.zeros((h,w,3),dtype=np.uint8)
        for c in range(3):
            ch=arr[:,:,c].astype(np.float32)
            r[:,:,c]=(np.percentile(ch,10)+p*(np.percentile(ch,90)-np.percentile(ch,10))).clip(0,255).astype(np.uint8)
        arr=cv2.addWeighted(arr,1-a,r,a,0)
    elif effect == "weave":
        w2=int(spec.get("width",12)); g=int(spec.get("gap",2))
        r=np.zeros((h,w,3),dtype=np.uint8)
        for y in range(0,h,w2+g):
            py=(y//(w2+g))%2
            for x in range(0,w,w2+g):
                px=(x//(w2+g))%2; cy=min(y+w2//2,h-1); cx=min(x+w2//2,w-1)
                c=tuple(int(ci)for ci in arr[cy,cx])
                dark=tuple(int(ci*0.7)for ci in c)
                if py==px:
                    for dy in range(w2):
                        if y+dy<h: r[y+dy,x:min(x+w2,w)]=dark
                else:
                    for dx in range(w2):
                        if x+dx<w:
                            for dy in range(w2):
                                if y+dy<h: r[y+dy,x+dx]=c
        arr=r
    elif effect == "oil_spill":
        i=float(spec.get("intensity",0.5)); sc=float(spec.get("scale",0.03))
        yy,xx=np.mgrid[:h,:w].astype(np.float32)
        g=cv2.cvtColor(arr,cv2.COLOR_RGB2GRAY).astype(np.float32)
        gx=cv2.Sobel(g,cv2.CV_32F,1,0,ksize=3); gy=cv2.Sobel(g,cv2.CV_32F,0,1,ksize=3)
        gr=np.sqrt(gx**2+gy**2); gr=gr/(gr.max()+1e-8)
        ph=xx*sc*5+yy*sc*3+gr*2
        ir=np.stack([((np.sin(ph)+1)*0.5).clip(0,1),((np.sin(ph+2.09)+1)*0.5).clip(0,1),((np.sin(ph+4.19)+1)*0.5).clip(0,1)],axis=-1).astype(np.float32)
        arr=(arr.astype(np.float32)*(1-i)+ir*255*i).clip(0,255).astype(np.uint8)
    elif effect == "bubbles":
        d=int(spec.get("density",15)); random.seed(13); r=arr.astype(np.float32)
        for _ in range(d):
            cx=random.randint(w//6,5*w//6); cy=random.randint(h//6,5*h//6); rad=random.randint(15,60)
            yy,xx=np.mgrid[:h,:w].astype(np.float32); dist=np.sqrt((xx-cx)**2+(yy-cy)**2); m=dist<rad
            if not m.any(): continue
            fp=dist*0.3+yy*0.05+xx*0.03
            rf,bf,gf=((np.sin(fp)+1)*0.5).clip(0,1),((np.sin(fp+2.09)+1)*0.5).clip(0,1),((np.sin(fp+4.19)+1)*0.5).clip(0,1)
            a=float(spec.get("alpha",0.4)); eb=(dist>rad-3)&(dist<=rad)
            for c in range(3): r[m,c]=r[m,c]*(1-a)+255*[rf[m],bf[m],gf[m]][c]*a
            r[eb]+=np.array([40,40,60])
        arr=r.clip(0,255).astype(np.uint8)
    elif effect == "crystallize":
        cs=int(spec.get("cell_size",20)); random.seed(int(spec.get("seed",42)))
        pts=[(np.clip(gx*cs+random.randint(-cs//3,cs//3),0,w-1),np.clip(gy*cs+random.randint(-cs//3,cs//3),0,h-1)) for gy in range(0,h,cs) for gx in range(0,w,cs)]
        for _ in range(int(len(pts)*0.3)): pts.append((random.randint(0,w-1),random.randint(0,h-1)))
        yy,xx=np.mgrid[:h,:w]; dists=np.zeros((h,w,len(pts)))
        for i,(px,py) in enumerate(pts): dists[:,:,i]=(xx-px)**2+(yy-py)**2
        n=np.argmin(dists,axis=2); r=np.zeros_like(arr)
        for i,(px,py) in enumerate(pts): m=n==i; r[m]=arr[py,px]if m.any()else r[m]
        e=cv2.Canny(cv2.cvtColor(r,cv2.COLOR_RGB2GRAY),5,20); r[e>0]=(0,0,0); arr=r
    elif effect == "voronoi_stipple":
        dr=int(spec.get("dot_radius",3)); d=float(spec.get("density",0.6)); random.seed(42)
        g=cv2.cvtColor(arr,cv2.COLOR_RGB2GRAY).astype(np.float32)/255
        pts=[(random.randint(0,w-1),random.randint(0,h-1)) for _ in range(int(d*8000)) if random.random()>g[random.randint(0,h-1),random.randint(0,w-1)]]
        if len(pts)<10: pts=[(random.randint(0,w-1),random.randint(0,h-1))for _ in range(50)]
        yy,xx=np.mgrid[:h,:w]; dists=np.zeros((h,w,len(pts)))
        for i,(px,py) in enumerate(pts): dists[:,:,i]=(xx-px)**2+(yy-py)**2
        n=np.argmin(dists,axis=2); r=np.ones((h,w,3),dtype=np.uint8)*255
        for i,(px,py) in enumerate(pts):
            cy, cx=np.clip(py,0,h-1),np.clip(px,0,w-1)
            cv2.circle(r,(cx,cy),max(1,int(dr*(1-g[cy,cx])*2)),tuple(int(ci)for ci in arr[cy,cx]),-1)
        arr=r
    elif effect == "cross_stitch":
        st=int(spec.get("stitch",8)); r=np.ones((h,w,3),dtype=np.uint8)*245
        if spec.get("fabric",True):
            for y in range(0,h,st): r[y:min(y+1,h)]=(r[y:min(y+1,h)].astype(np.float32)*0.9).astype(np.uint8)
            for x in range(0,w,st): r[:,x:min(x+1,w)]=(r[:,x:min(x+1,w)].astype(np.float32)*0.9).astype(np.uint8)
        for y in range(0,h-st,st):
            for x in range(0,w-st,st):
                cy=min(y+st//2,h-1); cx=min(x+st//2,w-1); c=tuple(int(ci)for ci in arr[cy,cx])
                cv2.line(r,(x,y),(x+st-1,y+st-1),c,max(1,st//4)); cv2.line(r,(x+st-1,y),(x,y+st-1),c,max(1,st//4))
        arr=r
    elif effect == "aurora":
        i=float(spec.get("intensity",0.5))
        yy,xx=np.mgrid[:h,:w].astype(np.float32); ph=xx*0.05+np.sin(yy*0.02)*2
        c=np.sin(ph+yy*0.03)
        ga=np.exp(-((c-0.6)/0.3)**2)*np.exp(-yy/(h*0.4))
        ra=np.exp(-((c-0.2)/0.2)**2)*np.exp(-yy/(h*0.3))
        ba=np.exp(-((c+0.4)/0.25)**2)*np.exp(-yy/(h*0.5))
        au=np.stack([ra*20+ga*10,ga*60+ba*10,ba*80+ra*5],axis=-1).clip(0,255).astype(np.uint8)
        au=cv2.GaussianBlur(au.astype(np.float32),(0,0),sigmaX=8)
        arr=cv2.addWeighted(arr.astype(np.float32),1,au,i,0).clip(0,255).astype(np.uint8)
    elif effect == "lightning":
        n=int(spec.get("num_bolts",3)); random.seed(17)
        o=np.zeros((h,w,3),dtype=np.uint8)
        for _ in range(n):
            x,y=random.randint(w//4,3*w//4),0; pts=[(x,y)]
            while y<h:
                x+=random.randint(-15,15); y+=random.randint(10,30)
                pts.append((np.clip(x,0,w-1),np.clip(y,0,h-1)))
                if random.random()<0.2:
                    bx,by=pts[-1]
                    for _ in range(random.randint(3,8)):
                        bx+=random.randint(-10,10); by+=random.randint(5,15)
                        cv2.line(o,pts[-1],(np.clip(bx,0,w-1),np.clip(by,0,h-1)),(200,220,255),random.randint(1,2))
            for i in range(len(pts)-1): cv2.line(o,pts[i],pts[i+1],(200,220,255),2)
        o=cv2.GaussianBlur(o.astype(np.float32),(0,0),sigmaX=4)
        arr=cv2.addWeighted(arr.astype(np.float32),1,o,float(spec.get("alpha",0.6)),0).clip(0,255).astype(np.uint8)
    elif effect == "plasma":
        sc=float(spec.get("scale",0.02)); i=float(spec.get("intensity",0.5)); b=spec.get("blend","screen")
        yy,xx=np.mgrid[:h,:w].astype(np.float32)
        p=(np.sin(xx*sc+yy*sc*0.5)*0.5+np.sin(xx*sc*2+yy*sc*1.5)*0.25+np.sin(xx*sc*4+yy*sc*3)*0.125+np.sin(xx*sc*8+yy*sc*6)*0.0625)
        p=(p+1)/2; r=(np.sin(p*6.28)*0.5+0.5).clip(0,1); g=(np.sin(p*6.28+2.09)*0.5+0.5).clip(0,1); b2=(np.sin(p*6.28+4.19)*0.5+0.5).clip(0,1)
        pl=np.stack([r,g,b2],axis=-1)
        if b=="screen": arr=(255-(255-arr.astype(np.float32))*(255-pl*255)/255).clip(0,255).astype(np.uint8)
        elif b=="add": arr=(arr.astype(np.float32)+pl*255*i).clip(0,255).astype(np.uint8)
    elif effect == "melting":
        a=float(spec.get("amount",0.3)); g=cv2.cvtColor(arr,cv2.COLOR_RGB2GRAY).astype(np.float32)
        r=np.zeros_like(arr)
        for x in range(w):
            cm=g[:,x].mean()/255; dr=int(a*cm*h*0.3)
            if dr>0: r[:,x]=np.roll(arr[:,x],dr,0); r[:dr,x]=arr[min(dr,h-1),x]
            else: r[:,x]=arr[:,x]
        for x in range(w):
            dl=int(a*g[:,x].mean()/255*30)
            for d in range(dl):
                ly=h-1+d
                if ly<h: r[ly,x]=arr[min(h-1-d,h-1),x]
        arr=cv2.GaussianBlur(r.astype(np.float32),(3,3),0).clip(0,255).astype(np.uint8)
    elif effect == "liquify":
        s=float(spec.get("strength",0.4)); random.seed(42)
        yy,xx=np.mgrid[:h,:w].astype(np.float32)
        fx=cv2.GaussianBlur(np.random.randn(h,w).astype(np.float32),(0,0),sigmaX=30)*s*40
        fy=cv2.GaussianBlur(np.random.randn(h,w).astype(np.float32),(0,0),sigmaX=30)*s*40
        arr=cv2.remap(arr,np.clip(xx+fx,0,w-1).astype(np.float32),np.clip(yy+fy,0,h-1).astype(np.float32),cv2.INTER_LINEAR)
    elif effect == "shatter":
        n=int(spec.get("cracks",20)); random.seed(19); r=arr.copy()
        ix,iy=random.randint(w//4,3*w//4),random.randint(h//4,3*h//4)
        for _ in range(n):
            a=random.uniform(0,math.pi*2); l=random.randint(20,120)
            pts=[(ix,iy)]; cx,cy=float(ix),float(iy)
            steps=max(10,l//5)
            for s in range(steps):
                cx+=(ix+int(l*math.cos(a))-ix)/steps+random.randint(-5,5)
                cy+=(iy+int(l*math.sin(a))-iy)/steps+random.randint(-5,5)
                pts.append((np.clip(int(cx),0,w-1),np.clip(int(cy),0,h-1)))
            for i in range(len(pts)-1): cv2.line(r,pts[i],pts[i+1],(0,0,0),1)
        cm=(cv2.cvtColor(r,cv2.COLOR_RGB2GRAY)<50).astype(np.uint8)*255
        cg=cv2.GaussianBlur(cm.astype(np.float32),(5,5),0)*0.3
        for c in range(3): r[:,:,c]=(r[:,:,c].astype(np.float32)+cg*30).clip(0,255).astype(np.uint8)
        arr=r
    elif effect == "hatching":
        sp=int(spec.get("spacing",8)); a=float(spec.get("angle",45))
        g=cv2.cvtColor(arr,cv2.COLOR_RGB2GRAY).astype(np.float32); r=np.ones((h,w,3),dtype=np.uint8)*255
        rad=math.radians(a); ca,sa=math.cos(rad),math.sin(rad)
        for i in range(-int(np.sqrt(w**2+h**2)),int(np.sqrt(w**2+h**2)),sp):
            x0,y0=int(i*ca),int(i*sa)
            for t in range(0,max(w,h)):
                sx,sy=int(x0+t*ca),int(y0+t*sa)
                if 0<=sx<w and 0<=sy<h: r[sy,sx]=tuple(int(ci*(1-g[sy,sx]/255*0.8))for ci in r[sy,sx])
        arr=r
    elif effect == "conway":
        gen=int(spec.get("generations",5)); cs=int(spec.get("cell_size",4))
        g=cv2.cvtColor(arr,cv2.COLOR_RGB2GRAY); gh, gw=h//cs, w//cs
        if gh>=3 and gw>=3:
            c2=np.zeros((gh,gw),dtype=np.uint8)
            for gy in range(gh):
                for gx in range(gw):
                    c2[gy,gx]=1 if g[gy*cs:(gy+1)*cs,gx*cs:(gx+1)*cs].mean()>128 else 0
            for _ in range(gen):
                p2=np.pad(c2,1,mode='constant')
                nb=p2[:-2,:-2]+p2[:-2,1:-1]+p2[:-2,2:]+p2[1:-1,:-2]+p2[1:-1,2:]+p2[2:,:-2]+p2[2:,1:-1]+p2[2:,2:]
                c2=((c2==1)&((nb==2)|(nb==3)))|((c2==0)&(nb==3)); c2=c2.astype(np.uint8)
            r=np.ones((h,w,3),dtype=np.uint8)*255
            for gy in range(gh):
                for gx in range(gw):
                    if c2[gy,gx]:
                        y0,y1=gy*cs,min((gy+1)*cs,h); x0,x1=gx*cs,min((gx+1)*cs,w)
                        r[y0:y1,x0:x1]=tuple(int(ci)for ci in arr[(y0+y1)//2,(x0+x1)//2])
            arr=r
    elif effect == "neon_sign":
        th=int(spec.get("threshold",60)); c=spec.get("color",(255,80,200))
        if isinstance(c,str): c=_parse_color(c)
        g=cv2.cvtColor(arr,cv2.COLOR_RGB2GRAY); b2=(g>th).astype(np.uint8)
        core=cv2.erode(b2,np.ones((3,3),dtype=np.uint8),iterations=1)
        gl=cv2.GaussianBlur(b2.astype(np.float32),(15,15),0)
        rn=np.zeros((h,w,3),dtype=np.float32)
        for c2 in range(3): rn[:,:,c2]=c[c2]*gl*float(spec.get("glow_intensity",0.6))
        rn=np.maximum(rn,np.stack([core,core,core],axis=-1).astype(np.float32)*255*0.8)
        arr=rn.clip(0,255).astype(np.uint8)
    elif effect == "colorize":
        pl=spec.get("palette",[(20,20,100),(80,80,200),(200,100,80),(255,230,180)])
        if isinstance(pl[0],str): pl=[_parse_color(c)for c in pl]
        g=cv2.cvtColor(arr,cv2.COLOR_RGB2GRAY).astype(np.float32)/255
        r=np.zeros((*g.shape,3),dtype=np.uint8)
        for c in range(3): r[:,:,c]=np.interp(g,np.linspace(0,1,len(pl)),[p[c]for p in pl]).clip(0,255).astype(np.uint8)
        arr=cv2.addWeighted(arr,1-float(spec.get("alpha",1)),r,float(spec.get("alpha",1)),0)

    elif effect == "pastel":
        b=int(spec.get("blur",9)); b+=1 if b%2==0 else 0
        s=cv2.GaussianBlur(arr,(b,b),0).astype(np.float32); g=cv2.cvtColor(arr,cv2.COLOR_RGB2GRAY).astype(np.float32)/255
        p=s*0.7+g[:,:,None]*80*0.3+30+np.random.rand(h,w,3).astype(np.float32)*15*(1-g[:,:,None]*0.5)
        arr=p.clip(0,255).astype(np.uint8)
    elif effect == "watercolor_wash":
        s=float(spec.get("strength",0.7))
        b=cv2.GaussianBlur(arr,(25,25),0).astype(np.float32)
        hsv=cv2.cvtColor(b.clip(0,255).astype(np.uint8),cv2.COLOR_RGB2HSV).astype(np.float32)
        hsv[:,:,1]*=1.3; hsv[:,:,2]*=0.8
        b=cv2.cvtColor(hsv.clip(0,255).astype(np.uint8),cv2.COLOR_HSV2RGB).astype(np.float32)
        b+=np.random.randn(h,w).astype(np.float32)[:,:,None]*8
        yy,xx=np.mgrid[:h,:w].astype(np.float32); b+=np.sin(xx*0.01+yy*0.02)[:,:,None]*10
        arr=cv2.addWeighted(arr.astype(np.float32),1-s,b.astype(np.float32),s,0).clip(0,255).astype(np.uint8)
    elif effect == "gouache":
        b=int(spec.get("blur",11)); b+=1 if b%2==0 else 0; s=float(spec.get("strength",0.8))
        f=cv2.bilateralFilter(cv2.medianBlur(arr,9),b,40,40).astype(np.float32); f=f*0.85+20
        f+=np.random.randn(h,w,3).astype(np.float32)*5
        arr=cv2.addWeighted(arr.astype(np.float32),1-s,f.astype(np.float32),s,0).clip(0,255).astype(np.uint8)
    elif effect == "colored_pencil":
        g=cv2.cvtColor(arr,cv2.COLOR_RGB2GRAY).astype(np.float32)/255; r=np.zeros((h,w,3),dtype=np.float32); random.seed(42)
        for l in range(3):
            a2=l*60; rad=math.radians(a2); sp=4+l*2
            yy,xx=np.mgrid[:h,:w].astype(np.float32)
            h2=((np.sin((xx*math.cos(rad)+yy*math.sin(rad))/sp)+1)/2)
            ls2=0.4+l*0.1
            for c in range(3): r[:,:,c]+=h2*(1-g)*ls2*(arr[:,:,c].astype(np.float32)/255)
        arr=((1-r)*255).clip(0,255).astype(np.uint8)
    elif effect == "cel_animation":
        b=int(spec.get("blur",7)); b+=1 if b%2==0 else 0; l=int(spec.get("levels",8)); st=255//(l-1)
        f=cv2.pyrMeanShiftFiltering(arr,10,30).astype(np.float32); f=(f//st*st+st//2).clip(0,255).astype(np.uint8)
        e=cv2.dilate(cv2.Canny(cv2.cvtColor(arr,cv2.COLOR_RGB2GRAY),20,60),np.ones((2,2),dtype=np.uint8),iterations=1)
        f[e>0]=(0,0,0)
        bg=spec.get("bg",(240,240,255))
        if isinstance(bg,str): bg=_parse_color(bg)
        arr=cv2.addWeighted(f,0.9,np.full_like(f,bg,dtype=np.uint8),0.1,0)
    elif effect == "storyboard":
        g=cv2.cvtColor(arr,cv2.COLOR_RGB2GRAY).astype(np.float32)
        e=cv2.dilate(cv2.Canny(arr,30,100),np.ones((2,2),dtype=np.uint8),iterations=1)
        r=np.ones((h,w,3),dtype=np.uint8)*245; r[:,:,0]=(r[:,:,0].astype(np.float32)*0.95).astype(np.uint8); r[:,:,2]=(r[:,:,2].astype(np.float32)*0.9).astype(np.uint8)
        r[e>0]=(60,100,180)
        s2=(g<80).astype(np.uint8); s2=cv2.erode(s2,np.ones((2,2),dtype=np.uint8),iterations=1)
        for y in range(0,h,6):
            for x in range(0,w,12):
                if s2[min(y,h-1),min(x,w-1)]: cv2.line(r,(x,y),(x+8,y),(60,100,180),1)
        cv2.rectangle(r,(5,5),(w-5,h-5),(40,60,120),2)
        cv2.putText(r,f"SC {int(spec.get('scene',1)):02d}",(15,h-15),cv2.FONT_HERSHEY_SIMPLEX,0.5,(40,60,120),1)
        arr=r
    elif effect == "lens_flare_anamorphic":
        i=float(spec.get("intensity",0.5)); o=np.zeros((h,w,3),dtype=np.float32)
        b2=(cv2.cvtColor(arr,cv2.COLOR_RGB2GRAY)>200).astype(np.uint8)
        b2=cv2.dilate(b2,np.ones((5,5),dtype=np.uint8),iterations=1)
        nl,_,_,ct=cv2.connectedComponentsWithStats(b2,connectivity=8)
        for i2 in range(1,nl):
            cx,cy=int(ct[i2][0]),int(ct[i2][1]); sl=int(w*0.3*i)
            for t in range(-sl,sl):
                xt=cx+t
                if 0<=xt<w: af=1-abs(t)/sl; o[cy,xt]+=np.array([255*af*0.8,200*af*0.5,100*af*0.2])
            for t in range(-sl//2,sl//2):
                xt=cx+t; yt=min(cy+20,h-1)
                if 0<=xt<w: af=1-abs(t)/(sl//2); o[yt,xt]+=np.array([200*af*0.3,220*af*0.5,255*af*0.8])
        o=cv2.GaussianBlur(o,(5,5),0)
        arr=cv2.addWeighted(arr.astype(np.float32),1,o,1,0).clip(0,255).astype(np.uint8)
    elif effect == "dutch_tilt":
        a=float(spec.get("angle",5))
        arr=cv2.warpAffine(arr,cv2.getRotationMatrix2D((w/2,h/2),a,1),(w,h),borderMode=cv2.BORDER_REPLICATE)
    elif effect == "vignette_color":
        c=spec.get("color",(180,80,100))
        if isinstance(c,str): c=_parse_color(c)
        s=float(spec.get("strength",0.4))
        yy,xx=np.mgrid[:h,:w]; d=np.sqrt((xx-w/2)**2+(yy-h/2)**2)/np.sqrt((w/2)**2+(h/2)**2); m=d**2
        r=arr.astype(np.float32)
        for c2 in range(3): r[:,:,c2]=r[:,:,c2]*(1-m*s*0.5)+c[c2]*m*s*0.5
        arr=r.clip(0,255).astype(np.uint8)
    elif effect == "chromatic_focus":
        fx=int(spec.get("focal_x",w//2)); fy=int(spec.get("focal_y",h//2)); ms=int(spec.get("max_shift",8))
        r,g,b=cv2.split(arr.astype(np.float32))
        yy,xx=np.mgrid[:h,:w].astype(np.float32)
        d=np.sqrt((xx-fx)**2+(yy-fy)**2)/np.sqrt((w/2)**2+(h/2)**2)
        sm=(d*ms).astype(np.int32)
        for y in range(h):
            for x in range(w):
                if sm[y,x]>0: r[y,x]=arr[y,min(x+sm[y,x],w-1),0].astype(np.float32); b[y,x]=arr[y,max(x-sm[y,x],0),2].astype(np.float32)
        arr=cv2.merge([r,g,b]).clip(0,255).astype(np.uint8)
    elif effect == "c41":
        s=float(spec.get("strength",0.7))
        r=arr[:,:,0].astype(np.float32)*0.8+50; g=arr[:,:,1].astype(np.float32)*0.6+80; b=arr[:,:,2].astype(np.float32)*0.4+120
        ci=np.stack([r,g,b],axis=-1)
        gc=cv2.cvtColor(ci.clip(0,255).astype(np.uint8),cv2.COLOR_RGB2GRAY).astype(np.float32)+15
        ci=cv2.addWeighted(ci.clip(0,255).astype(np.uint8),0.8,cv2.cvtColor(gc.clip(0,255).astype(np.uint8),cv2.COLOR_GRAY2RGB),0.2,0)
        arr=cv2.addWeighted(arr,1-s,ci,s,0)
    elif effect == "ektachrome":
        s=float(spec.get("strength",0.6))
        r=arr[:,:,0].astype(np.float32)*1.1; g=arr[:,:,1].astype(np.float32)*0.9; b=arr[:,:,2].astype(np.float32)*1.3
        ek=np.stack([r,g,b],axis=-1)
        lab=cv2.cvtColor(ek.clip(0,255).astype(np.uint8),cv2.COLOR_RGB2LAB).astype(np.float32); l,a2,b2=cv2.split(lab)
        l=np.clip((l-50)*1.3+50,0,255)
        ek=cv2.cvtColor(cv2.merge([l,a2,b2]).clip(0,255).astype(np.uint8),cv2.COLOR_LAB2RGB)
        sm=1-cv2.cvtColor(ek,cv2.COLOR_RGB2GRAY).astype(np.float32)/255
        for c in range(3): ek[:,:,c]=ek[:,:,c].astype(np.float32)+sm*(20 if c==0 else 0)
        arr=cv2.addWeighted(arr,1-s,ek.clip(0,255).astype(np.uint8),s,0)
    elif effect == "kodachrome":
        s=float(spec.get("strength",0.7))
        r=arr[:,:,0].astype(np.float32)*1.2; g=arr[:,:,1].astype(np.float32)*0.95; b=arr[:,:,2].astype(np.float32)*0.85
        kc=np.clip((np.stack([r,g,b],axis=-1)/255-0.05)*1.2+0.05,0,1)*255+np.random.randn(h,w,3).astype(np.float32)*6
        arr=cv2.addWeighted(arr,1-s,kc.clip(0,255).astype(np.uint8),s,0)
    elif effect == "d7000_banding":
        s=float(spec.get("strength",0.5)); g=cv2.cvtColor(arr,cv2.COLOR_RGB2GRAY).astype(np.float32)/255
        sd=g<0.3; bd=np.zeros((h,w,3),dtype=np.float32)
        for y in range(h):
            b2=0.5+0.5*np.sin(y*0.2+y*0.01); bd[y,:,0]=b2*15; bd[y,:,2]=(1-b2)*15
        bd*=sd[:,:,None].astype(np.float32)
        arr=cv2.addWeighted(arr.astype(np.float32),1,(arr.astype(np.float32)+bd).clip(0,255),0.5,0).clip(0,255).astype(np.uint8)
    elif effect == "glitch_vhs_tracking":
        s=float(spec.get("severity",0.4)); random.seed(42); r=arr.copy()
        for _ in range(int(s*15)):
            y=random.randint(0,h-10); bh=min(random.randint(3,20),h-y)
            sh2=random.randint(-30,30)
            if sh2!=0: r[y:y+bh]=np.roll(r[y:y+bh],sh2,axis=1)
            if random.random()<0.3:
                nb=np.random.randint(0,255,(bh,w,3),dtype=np.uint8)
                r[y:y+bh]=cv2.addWeighted(r[y:y+bh].astype(np.float32),1-random.uniform(0.1,0.5),nb.astype(np.float32),random.uniform(0.1,0.5),0).clip(0,255).astype(np.uint8)
        tj=random.randint(-5,5); jh=random.randint(5,30)
        r[:jh]=np.roll(r[:jh],tj,axis=1); arr=r
    elif effect == "datamosh":
        s=float(spec.get("severity",0.3)); bs=int(spec.get("block_size",16)); random.seed(42); r=arr.copy()
        for _ in range(int(s*30)):
            bx=random.randint(0,w-bs); by=random.randint(0,h-bs)
            bw2=min(bs,w-bx); bh2=min(bs,h-by)
            sx=random.randint(0,w-bw2); sy=random.randint(0,h-bh2)
            r[by:by+bh2,bx:bx+bw2]=arr[sy:sy+bh2,sx:sx+bw2]
        arr=r
    elif effect == "echo":
        n=int(spec.get("echoes",5)); d=float(spec.get("decay",0.6)); dx=int(spec.get("delay_x",15)); dy=int(spec.get("delay_y",0))
        r=np.zeros_like(arr,dtype=np.float32)
        for i in range(n):
            w2=d**i; ox=dx*i; oy=dy*i
            c2=np.roll(arr.astype(np.float32),oy,0); c2=np.roll(c2,ox,1)
            if ox>0: c2[:,:ox]=0
            elif ox<0: c2[:,ox:]=0
            if oy>0: c2[:oy]=0
            elif oy<0: c2[oy:]=0
            r+=c2*w2
        arr=r.clip(0,255).astype(np.uint8)
    elif effect == "auto_painterly":
        s=cv2.pyrMeanShiftFiltering(arr,15,40).astype(np.float32)
        e=cv2.dilate(cv2.Canny(cv2.cvtColor(arr,cv2.COLOR_RGB2GRAY),30,80),np.ones((2,2),dtype=np.uint8),iterations=1)
        t2=np.ones((h,w,3),dtype=np.float32)*248
        for y2 in range(0,h,4):
            wv=0.95+np.sin(np.arange(w)*0.3)*0.03; t2[y2:min(y2+2,h)]*=wv[None,:,None]
        r=s*0.85+t2*0.15; r[e>0]=arr[e>0].astype(np.float32)*1.2
        arr=r.clip(0,255).astype(np.uint8)
    elif effect == "time_lapse_day_for_night":
        s=float(spec.get("strength",0.8)); a=arr.astype(np.float32)*(0.3+0.3*(1-s))
        a[:,:,0]*=0.7; a[:,:,2]*=1.4
        g=cv2.cvtColor(arr,cv2.COLOR_RGB2GRAY).astype(np.float32)/255
        st2=(np.random.rand(h,w)>0.995).astype(np.float32)*255
        st2=cv2.GaussianBlur(st2,(3,3),0)
        a[:,:,0]+=st2*0.5; a[:,:,1]+=st2*0.5; a[:,:,2]+=st2
        arr=np.clip((a/255-0.1)*1.5+0.1,0,1)*255; arr=arr.clip(0,255).astype(np.uint8)
    elif effect == "pixel_warhol":
        sw=spec.get("style","pop")
        panels=[spec.get(f"color{i+1}",[("cyan","magenta"),("yellow","blue"),("red","green"),("orange","purple")][i])for i in range(4)]
        parsed=[]
        for p in panels: parsed.append((_parse_color(p[0])if isinstance(p[0],str)else tuple(p[0]),_parse_color(p[1])if isinstance(p[1],str)else tuple(p[1])))
        g=cv2.cvtColor(arr,cv2.COLOR_RGB2GRAY).astype(np.float32)/255
        hh,hw=h//2,w//2; r=np.zeros((h,w,3),dtype=np.uint8)
        for idx,(c1,c2)in enumerate(parsed[:4]):
            px,py=idx%2,idx//2; x0,y0=0 if px==0 else hw,0 if py==0 else hh
            x1,y1=min(x0+hw,w),min(y0+hh,h); pg=g[y0:y1,x0:x1]
            pi=np.zeros((y1-y0,x1-x0,3),dtype=np.uint8)
            for c in range(3): pi[:,:,c]=((c1[c]*(1-pg)+c2[c]*pg)*255).clip(0,255).astype(np.uint8)
            if sw=="pop":
                e2=cv2.dilate(cv2.Canny(cv2.cvtColor(pi,cv2.COLOR_RGB2GRAY),30,80),np.ones((2,2),dtype=np.uint8),iterations=1)
                pi[e2>0]=(0,0,0)
            r[y0:y1,x0:x1]=pi
        arr=r
    elif effect == "noise_crt_interference":
        i=float(spec.get("intensity",0.5))
        yy,xx=np.mgrid[:h,:w].astype(np.float32)
        by=(yy+np.sin(xx*0.05)*10)%30<5
        be=by.astype(np.float32)*60
        nn=np.random.randn(h,w).astype(np.float32)*20*i
        o=np.zeros_like(arr,dtype=np.float32); o[:,:,0]+=be+nn*0.5; o[:,:,1]+=nn*0.3; o[:,:,2]+=nn*0.7
        arr=cv2.addWeighted(arr.astype(np.float32),1,o,1,0).clip(0,255).astype(np.uint8)
    elif effect == "perspective":
        ts=float(spec.get("top_shift",0.1))*w; bs=float(spec.get("bottom_shift",-0.1))*w
        src=np.float32([[0,0],[w,0],[w,h],[0,h]])
        dst=np.float32([[ts,0],[w+ts,0],[w+bs,h],[bs,h]])
        arr=cv2.warpPerspective(arr,cv2.getPerspectiveTransform(src,dst),(w,h),borderMode=cv2.BORDER_REPLICATE)
    elif effect == "panorama":
        fov=math.radians(float(spec.get("fov",120)))
        yy,xx=np.mgrid[:h,:w].astype(np.float32)
        theta=(xx-w/2)/(w/2)*fov/2
        src_x=np.clip(w/2+w/2*np.sin(theta)/(fov/2),0,w-1).astype(np.float32)
        src_y=np.clip(yy+(yy-h/2)*(1-np.cos(theta))*0.3,0,h-1).astype(np.float32)
        arr=cv2.remap(arr,src_x,src_y,cv2.INTER_LINEAR)
    elif effect == "rainbow_arc":
        i=float(spec.get("intensity",0.4)); cx,cy=w/2,h*1.2; rmn=h*float(spec.get("radius_min",0.6)); rmx=h*float(spec.get("radius_max",0.8))
        yy,xx=np.mgrid[:h,:w].astype(np.float32); d=np.sqrt((xx-cx)**2+(yy-cy)**2)
        a2=np.arctan2(yy-cy,xx-cx); ua=a2>-math.pi*0.8
        ib=(d>rmn)&(d<rmx)&ua; tv=((d-rmn)/(rmx-rmn)).clip(0,1)
        ri=np.zeros((h,w),dtype=np.float32); gi=np.zeros((h,w),dtype=np.float32); bi=np.zeros((h,w),dtype=np.float32)
        ri[ib]=((np.sin(tv[ib]*6.28)*0.5+0.5)*255).clip(0,255); gi[ib]=((np.sin(tv[ib]*6.28+2.09)*0.5+0.5)*255).clip(0,255); bi[ib]=((np.sin(tv[ib]*6.28+4.19)*0.5+0.5)*255).clip(0,255)
        rb=np.stack([ri,gi,bi],axis=-1).clip(0,255).astype(np.uint8)
        arr=cv2.addWeighted(arr.astype(np.float32),1,cv2.GaussianBlur(rb.astype(np.float32),(7,7),0),i,0).clip(0,255).astype(np.uint8)
    elif effect == "fire":
        i=float(spec.get("intensity",0.5)); random.seed(42)
        yy,xx=np.mgrid[:h,:w].astype(np.float32)
        fn=cv2.GaussianBlur(np.random.randn(h,w).astype(np.float32),(0,0),sigmaX=5)
        fh=1-yy/h; fl=np.sin(xx*0.05+fh*5+fn*2)*0.5+0.5; fl=(fl*fh).clip(0,1)
        fi2=np.stack([(fl*255).clip(0,255).astype(np.uint8),(fl*180*(1-fl*0.5)).clip(0,255).astype(np.uint8),(fl*60*(1-fl*0.8)).clip(0,255).astype(np.uint8)],axis=-1)
        fi2=cv2.GaussianBlur(fi2.astype(np.float32),(5,5),0)
        arr=cv2.addWeighted(arr.astype(np.float32),1,fi2,i,0).clip(0,255).astype(np.uint8)
    elif effect == "nebula":
        i=float(spec.get("intensity",0.5)); random.seed(42)
        c1=cv2.GaussianBlur(np.random.randn(h,w).astype(np.float32),(0,0),sigmaX=40)
        c2=cv2.GaussianBlur(np.random.randn(h,w).astype(np.float32),(0,0),sigmaX=20)
        c3=cv2.GaussianBlur(np.random.randn(h,w).astype(np.float32),(0,0),sigmaX=10)
        np2=c1*0.5+c2*0.3+c3*0.2; np2=(np2-np2.min())/(np2.max()-np2.min()+1e-8)
        nb=np.stack([((np.sin(np2*4)*0.5+0.5)*255).clip(0,255).astype(np.uint8),((np.sin(np2*4+2.09)*0.5+0.5)*150).clip(0,255).astype(np.uint8),((np.sin(np2*4+4.19)*0.5+0.5)*255).clip(0,255).astype(np.uint8)],axis=-1)
        arr=cv2.addWeighted(arr.astype(np.float32),1,nb.astype(np.float32),i,0).clip(0,255).astype(np.uint8)
    elif effect == "refraction":
        s=float(spec.get("strength",0.3)); cx,cy=w/2,h/2
        yy,xx=np.mgrid[:h,:w].astype(np.float32); dx,dy=xx-cx,yy-cy; d=np.sqrt(dx**2+dy**2)/np.sqrt(cx**2+cy**2)
        dp=s*d**2
        arr=cv2.remap(arr,np.clip(cx+dx*(1-dp),0,w-1).astype(np.float32),np.clip(cy+dy*(1-dp),0,h-1).astype(np.float32),cv2.INTER_LINEAR)
    elif effect == "turbulence":
        s=float(spec.get("strength",0.5)); random.seed(42); nx2=int(spec.get("vortices",5))
        yy,xx=np.mgrid[:h,:w].astype(np.float32); fx=fy=np.zeros((h,w),dtype=np.float32)
        for _ in range(nx2):
            vx,vy=random.randint(0,w),random.randint(0,h); sv=random.uniform(5,20)*s
            dx,dy=xx-vx,yy-vy; d=np.sqrt(dx**2+dy**2)+1e-8
            fx+=-dy/d*sv; fy+=dx/d*sv
        fx=cv2.GaussianBlur(fx,(0,0),sigmaX=10); fy=cv2.GaussianBlur(fy,(0,0),sigmaX=10)
        arr=cv2.remap(arr,np.clip(xx+fx,0,w-1).astype(np.float32),np.clip(yy+fy,0,h-1).astype(np.float32),cv2.INTER_LINEAR)
    elif effect == "caustics":
        i=float(spec.get("intensity",0.4)); t2=float(spec.get("time",0))
        yy,xx=np.mgrid[:h,:w].astype(np.float32)
        c1=np.sin(xx*0.03+t2)*np.cos(yy*0.04+t2*0.7); c2=np.sin(xx*0.06-t2*1.3)*np.cos(yy*0.05+t2*0.5); c3=np.sin(xx*0.01+yy*0.02+t2*0.3)
        ca=(c1*0.5+c2*0.3+c3*0.2+1)*0.5; ca=ca**3
        ca2=np.stack([ca*200,ca*230,ca*255],axis=-1).clip(0,255).astype(np.uint8)
        arr=cv2.addWeighted(arr.astype(np.float32),1,cv2.GaussianBlur(ca2.astype(np.float32),(0,0),sigmaX=3),i,0).clip(0,255).astype(np.uint8)
    elif effect == "impressionism":
        ds=int(spec.get("dab_size",6)); random.seed(42)
        r=np.ones((h,w,3),dtype=np.uint8)*240
        for y in range(0,h,ds):
            for x in range(0,w,ds):
                sx=np.clip(x+random.randint(-2,2),0,w-1); sy=np.clip(y+random.randint(-2,2),0,h-1)
                c=tuple(int(ci)for ci in arr[sy,sx]); a2=random.uniform(0,math.pi)
                for t in range(-ds//3,ds//3+1):
                    px,py=np.clip(x+ds//2+int(t*math.cos(a2)),0,w-1),np.clip(y+ds//2+int(t*math.sin(a2)),0,h-1)
                    cv2.circle(r,(px,py),max(1,ds//4),c,-1)
        arr=r
    elif effect == "pointillist":
        sp2=int(spec.get("spacing",5)); r=np.ones((h,w,3),dtype=np.uint8)*255
        for y in range(0,h,sp2):
            for x in range(0,w,sp2):
                cv2.circle(r,(min(x,w-1),min(y,h-1)),max(1,int(sp2*float(spec.get("dot_radius",0.6))*0.5)),tuple(int(c)for c in arr[min(y,h-1),min(x,w-1)]),-1)
        arr=r
    elif effect == "fauvist":
        f=cv2.bilateralFilter(cv2.medianBlur(arr,9).astype(np.uint8),11,40,40).astype(np.float32)
        hsv=cv2.cvtColor(f.clip(0,255).astype(np.uint8),cv2.COLOR_RGB2HSV).astype(np.float32)
        hsv[:,:,1]*=1.6; hsv[:,:,2]*=1.1
        f=cv2.cvtColor(hsv.clip(0,255).astype(np.uint8),cv2.COLOR_HSV2RGB).astype(np.float32)
        yy,xx=np.mgrid[:h,:w].astype(np.float32); f=f+np.sin(xx*0.02+yy*0.01)[:,:,None]*10
        random.seed(42); bf=15
        for y in range(0,h,bf):
            for x in range(0,w,bf):
                if random.random()<0.3:
                    hs=random.randint(-15,15)
                    bhsv=cv2.cvtColor(f[y:min(y+bf,h),x:min(x+bf,w)].clip(0,255).astype(np.uint8),cv2.COLOR_RGB2HSV).astype(np.float32)
                    bhsv[:,:,0]=(bhsv[:,:,0]+hs)%180
                    f[y:min(y+bf,h),x:min(x+bf,w)]=cv2.cvtColor(bhsv.clip(0,255).astype(np.uint8),cv2.COLOR_HSV2RGB).astype(np.float32)
        arr=f.clip(0,255).astype(np.uint8)
    elif effect == "wet_plate":
        c2=float(spec.get("contrast",2.5)); g=cv2.cvtColor(arr,cv2.COLOR_RGB2GRAY).astype(np.float32)/255
        pl=np.clip((g-0.5)*c2+0.5,0,1)
        pi=np.stack([(pl*180).clip(0,255).astype(np.uint8),(pl*150).clip(0,255).astype(np.uint8),(pl*130).clip(0,255).astype(np.uint8)],axis=-1)
        yy,xx=np.mgrid[:h,:w].astype(np.float32); ed=np.minimum(np.minimum(xx,w-xx),np.minimum(yy,h-yy))
        ea=np.exp(-ed/30)*0.6
        for c in range(3): pi[:,:,c]=(pi[:,:,c].astype(np.float32)*(1-ea*0.5)+ea*200*(1-c*0.2)).clip(0,255).astype(np.uint8)
        random.seed(42)
        for _ in range(50): cv2.circle(pi,(random.randint(0,w-1),random.randint(0,h-1)),random.randint(2,8),(50,40,30),-1)
        arr=pi
    elif effect == "platinum_print":
        s=float(spec.get("strength",1)); g=cv2.cvtColor(arr,cv2.COLOR_RGB2GRAY).astype(np.float32)/255
        pv=np.clip(g*0.9+g**2*0.1,0,1)
        pi=np.stack([(pv*200+40).clip(0,255).astype(np.uint8),(pv*180+45).clip(0,255).astype(np.uint8),(pv*150+50).clip(0,255).astype(np.uint8)],axis=-1)
        pi=(pi.astype(np.float32)+np.random.rand(h,w,3).astype(np.float32)*5).clip(0,255).astype(np.uint8)
        arr=cv2.addWeighted(arr,1-s,pi,s,0)
    elif effect == "autochrome":
        ms=int(spec.get("mosaic",4)); yy,xx=np.mgrid[:h,:w]
        rm=(xx%(ms*2)<ms)&(yy%(ms*2)<ms); gm=((xx%(ms*2)>=ms)&(yy%(ms*2)<ms))|((xx%(ms*2)<ms)&(yy%(ms*2)>=ms)); bm=(xx%(ms*2)>=ms)&(yy%(ms*2)>=ms)
        r=np.zeros_like(arr); r[:,:,0]=np.where(rm,arr[:,:,0],0); r[:,:,1]=np.where(gm,arr[:,:,1],0); r[:,:,2]=np.where(bm,arr[:,:,2],0)
        for c in range(3): r[:,:,c]=(r[:,:,c].astype(np.float32)*2.5).clip(0,255).astype(np.uint8)
        arr=cv2.GaussianBlur(r.astype(np.float32),(5,5),0).clip(0,255).astype(np.uint8)
    elif effect == "ambrotype":
        s=float(spec.get("strength",0.8)); g=cv2.cvtColor(arr,cv2.COLOR_RGB2GRAY).astype(np.float32)/255
        am=np.clip(g*0.6+g**3*0.4,0,1); am=1-am
        ai=np.stack([(am*60).clip(0,255).astype(np.uint8),(am*80).clip(0,255).astype(np.uint8),(am*120).clip(0,255).astype(np.uint8)],axis=-1)
        yy,xx=np.mgrid[:h,:w].astype(np.float32); ai=(ai.astype(np.float32)+np.sin(xx*0.01)[:,:,None]*10).clip(0,255).astype(np.uint8)
        arr=cv2.addWeighted(arr,1-s,ai,s,0)
    elif effect == "felt":
        s=cv2.GaussianBlur(arr,(11,11),0).astype(np.float32)
        s=s*0.8+30+cv2.GaussianBlur(np.random.randn(h,w,3).astype(np.float32),(0,0),sigmaX=2)*15
        arr=s.clip(0,255).astype(np.uint8)
    elif effect == "velvet":
        av=arr.astype(np.float32)*0.7
        yy,xx=np.mgrid[:h,:w].astype(np.float32)
        nap=np.sin(yy*0.2+np.random.randn(h,w)*0.5); nap=(nap+10)/20
        for c in range(3): av[:,:,c]=av[:,:,c]*(0.8+nap*0.2)+5
        arr=av.clip(0,255).astype(np.uint8)
    elif effect == "corduroy":
        ws=int(spec.get("wale_spacing",8)); yy,xx=np.mgrid[:h,:w].astype(np.float32)
        rib=(xx%(ws*2)<ws).astype(np.float32); rp=(xx%(ws*2))/(ws*2); rs=0.6+0.4*np.sin(rp*math.pi)
        r2=arr.astype(np.float32)
        for c in range(3): r2[:,:,c]=r2[:,:,c]*rs*0.9+np.random.randn(h,w).astype(np.float32)*5
        arr=r2.clip(0,255).astype(np.uint8)
    elif effect == "tweed":
        yy,xx=np.mgrid[:h,:w].astype(np.float32)
        h2=((xx+yy*0.5)//8%2==0).astype(np.float32); h3=((xx-yy*0.5)//8%2==0).astype(np.float32); wv=h2*h3
        random.seed(42); sl=np.random.rand(h,w)>0.98
        r3=arr.astype(np.float32)
        for c in range(3): r3[:,:,c]=r3[:,:,c]*(0.7+wv*0.2); r3[sl,c]=r3[sl,c]*1.3
        r3+=np.random.randn(h,w).astype(np.float32)[:,:,None]*6
        arr=r3.clip(0,255).astype(np.uint8)
    elif effect == "stereoscopic":
        se=int(spec.get("separation",20)); hw2=w//2; g=cv2.cvtColor(arr,cv2.COLOR_RGB2GRAY).astype(np.float32)/255; dp=g*se
        r2=np.zeros((h,w,3),dtype=np.uint8)
        for y in range(h):
            for x in range(hw2):
                s2=int(dp[y,x]*0.5); r2[y,x]=arr[y,np.clip(x+s2,0,w-1)]
        for y in range(h):
            for x in range(hw2,w):
                s2=int(dp[y,x-hw2]*0.5); r2[y,x]=arr[y,np.clip(x-hw2-s2,0,w-1)]
        cv2.line(r2,(hw2,0),(hw2,h),(80,80,80),2); arr=r2
    elif effect == "glitch_tele":
        s=float(spec.get("severity",0.5)); r=arr.copy().astype(np.float32)
        for y in range(0,h,max(1,int(20/(s+0.1)))):
            bh=random.randint(1,4); sh2=random.randint(-10,10)
            if y+bh<h: r[y:y+bh]=np.roll(r[y:y+bh],sh2,axis=1)
        for _ in range(int(s*8)):
            y=random.randint(0,h-10); bh=random.randint(2,8)
            gc=cv2.cvtColor(r[y:min(y+bh,h)].clip(0,255).astype(np.uint8),cv2.COLOR_RGB2GRAY)
            r[y:min(y+bh,h)]=cv2.cvtColor(gc,cv2.COLOR_GRAY2RGB).astype(np.float32)
        bg2=max(4,int(16*(1-s)))
        for _ in range(int(s*5)):
            bx=random.randint(0,w-bg2); by=random.randint(0,h-bg2)
            bw2=min(bg2,w-bx); bh2=min(bg2,h-by)
            r[by:by+bh2,bx:bx+bw2]=r[by:by+bh2,bx:bx+bw2].mean(axis=(0,1))
        arr=r.clip(0,255).astype(np.uint8)
    elif effect == "time_warp":
        a=float(spec.get("amount",0.3))
        yy,xx=np.mgrid[:h,:w].astype(np.float32)
        off=a*30*np.sin(yy*0.02+xx*0.005)
        arr=cv2.remap(arr,np.clip(xx+off,0,w-1).astype(np.float32),yy.astype(np.float32),cv2.INTER_LINEAR)
    elif effect == "feedback":
        it=int(spec.get("iterations",3)); sc=float(spec.get("scale",0.9)); sh2=int(spec.get("shift",10))
        r=arr.astype(np.float32)
        for _ in range(it):
            nh2, nw2=int(h*sc),int(w*sc)
            if nh2<2 or nw2<2: break
            sm=cv2.resize(r.clip(0,255).astype(np.uint8),(nw2,nh2))
            sm=np.roll(sm,sh2,axis=1)
            cn=np.zeros((h,w,3),dtype=np.float32); xo,yo=(w-nw2)//2,(h-nh2)//2
            cn[yo:yo+nh2,xo:xo+nw2]=sm.astype(np.float32)
            r=cv2.addWeighted(r,0.5,cn,0.5,0)
        arr=r.clip(0,255).astype(np.uint8)
    elif effect == "edge_pixelate":
        bb=int(spec.get("base_block",16)); g=cv2.cvtColor(arr,cv2.COLOR_RGB2GRAY).astype(np.float32)
        e=cv2.Canny(arr,30,100); ed=cv2.GaussianBlur(e.astype(np.float32),(31,31),0); ed=ed/ed.max()
        r=np.zeros_like(arr)
        for y in range(0,h,2):
            for x in range(0,w,2):
                b2=max(2,int(bb*(1-ed[min(y,h-1),min(x,w-1)]*0.8)))
                bx0,by0=(x//b2)*b2,(y//b2)*b2; bx1,by1=min(bx0+b2,w),min(by0+b2,h)
                r[by0:by1,bx0:bx1]=arr[by0:by1,bx0:bx1].mean(axis=(0,1)).astype(np.uint8)
        arr=r
    elif effect == "rgb_split_motion":
        a=int(spec.get("amount",20)); d=float(spec.get("decay",0.6)); al=float(spec.get("alpha",1))
        r,g,b=cv2.split(arr.astype(np.float32))
        r=np.roll(r,a,1); r=np.roll(r,a//2,0); b=np.roll(b,-a,1); b=np.roll(b,-a//2,0)
        rr=np.stack([r*d+arr[:,:,0].astype(np.float32)*(1-d),g,b*d+arr[:,:,2].astype(np.float32)*(1-d)],axis=-1)
        arr=cv2.addWeighted(arr.astype(np.float32),1-al,rr,al,0).clip(0,255).astype(np.uint8)

    elif effect == "kernel":
        kt=spec.get("kernel","sharpen")
        ks={"sharpen":np.array([[0,-1,0],[-1,5,-1],[0,-1,0]],dtype=np.float32),"edge_detect":np.array([[-1,-1,-1],[-1,8,-1],[-1,-1,-1]],dtype=np.float32),"emboss_alt":np.array([[-2,-1,0],[-1,1,1],[0,1,2]],dtype=np.float32),"gaussian3":np.array([[1,2,1],[2,4,2],[1,2,1]],dtype=np.float32)/16,"gaussian5":np.array([[1,4,6,4,1],[4,16,24,16,4],[6,24,36,24,6],[4,16,24,16,4],[1,4,6,4,1]],dtype=np.float32)/256,"sobel_h":np.array([[-1,0,1],[-2,0,2],[-1,0,1]],dtype=np.float32),"sobel_v":np.array([[-1,-2,-1],[0,0,0],[1,2,1]],dtype=np.float32),"laplacian_3":np.array([[0,-1,0],[-1,4,-1],[0,-1,0]],dtype=np.float32),"laplacian_5":np.array([[-1,-1,-1,-1,-1],[-1,-1,-1,-1,-1],[-1,-1,24,-1,-1],[-1,-1,-1,-1,-1],[-1,-1,-1,-1,-1]],dtype=np.float32)/8,"mean_blur":np.ones((5,5),dtype=np.float32)/25,"motion_h":np.array([[0,0,0,0,0],[1,1,1,1,1],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0]],dtype=np.float32)/5,"motion_v":np.array([[0,1,0,0,0],[0,1,0,0,0],[0,1,0,0,0],[0,1,0,0,0],[0,1,0,0,0]],dtype=np.float32)/5,"comic":np.array([[-2,-1,0],[-1,2,1],[0,1,2]],dtype=np.float32),"invert_emboss":np.array([[2,1,0],[1,0,-1],[0,-1,-2]],dtype=np.float32)}
        i=float(spec.get("intensity",1)); r=cv2.filter2D(arr.astype(np.float32),-1,ks.get(kt,ks["sharpen"]))
        arr=cv2.addWeighted(arr.astype(np.float32),1-i,r,i,0).clip(0,255).astype(np.uint8) if i<1 else r.clip(0,255).astype(np.uint8)
    elif effect == "sabattier":
        s=float(spec.get("strength",0.7)); g=cv2.cvtColor(arr,cv2.COLOR_RGB2GRAY).astype(np.float32)/255
        sab=np.where((g>0.2)&(g<0.8),(g-0.5)*1.5+0.5,g)
        r=np.zeros_like(arr,dtype=np.float32)
        for c in range(3):
            ch=arr[:,:,c].astype(np.float32)/255
            sab_ch=np.interp(ch.flatten(),np.linspace(0,1,256),np.interp(np.linspace(0,1,256),g.flatten(),sab.flatten())).reshape(h,w)
            r[:,:,c]=ch*(1-s)+sab_ch*s
        arr=(r*255).clip(0,255).astype(np.uint8)
    elif effect == "reticulation":
        s=float(spec.get("strength",0.5)); random.seed(42)
        cr=cv2.GaussianBlur(np.random.randn(h,w).astype(np.float32),(0,0),sigmaX=8)
        cm=((np.abs(cr)>1.5).astype(np.float32)>0.3).astype(np.float32)
        r2=arr.astype(np.float32)
        for c in range(3): r2[:,:,c]=r2[:,:,c]*(1-cm*s*0.5)+255*cm*s*0.5*(1-c*0.3)
        yy,xx=np.mgrid[:h,:w].astype(np.float32); ed=np.minimum(np.minimum(xx,w-xx),np.minimum(yy,h-yy))
        curl=np.exp(-ed/50)*0.3
        for c in range(3): r2[:,:,c]=r2[:,:,c]*(1-curl*0.3)+220*curl*0.3
        arr=r2.clip(0,255).astype(np.uint8)
    elif effect == "selenium":
        s=float(spec.get("strength",1)); g=cv2.cvtColor(arr,cv2.COLOR_RGB2GRAY).astype(np.float32)/255
        se=np.stack([(g*180+50).clip(0,255).astype(np.uint8),(g*130+30).clip(0,255).astype(np.uint8),(g*100+60).clip(0,255).astype(np.uint8)],axis=-1)
        arr=cv2.addWeighted(arr,1-s,se,s,0)
    elif effect == "lith_print":
        c2=float(spec.get("contrast",2.5)); g2=float(spec.get("grain",0.3)); random.seed(42)
        g=cv2.cvtColor(arr,cv2.COLOR_RGB2GRAY).astype(np.float32)/255
        l=np.clip((g-0.5)*c2+0.5,0,1)
        lg=cv2.GaussianBlur(np.random.randn(h,w).astype(np.float32),(0,0),sigmaX=1.5)
        l=np.clip(l+lg*g2*0.1,0,1)
        arr=np.stack([(l*190+40).clip(0,255).astype(np.uint8),(l*140+30).clip(0,255).astype(np.uint8),(l*80+20).clip(0,255).astype(np.uint8)],axis=-1)
    elif effect == "thermal":
        g=cv2.cvtColor(arr,cv2.COLOR_RGB2GRAY).astype(np.float32)/255
        r=np.stack([(g*220+30).clip(0,255).astype(np.uint8),(g*180*(1-g)*4).clip(0,255).astype(np.uint8),((1-g)*200).clip(0,255).astype(np.uint8)],axis=-1)
        cv2.line(r,(0,h//2),(w,h//2),(80,80,80),1); cv2.line(r,(w//2,0),(w//2,h),(80,80,80),1)
        fs=max(0.3,min(w,h)/800)
        cv2.putText(r,"MAX",(10,30),cv2.FONT_HERSHEY_SIMPLEX,fs,(255,255,255),1); cv2.putText(r,"MIN",(10,h-10),cv2.FONT_HERSHEY_SIMPLEX,fs,(0,0,255),1)
        arr=r
    elif effect == "sonogram":
        g=cv2.cvtColor(arr,cv2.COLOR_RGB2GRAY).astype(np.float32)/255
        sp=int(spec.get("scan_pos",w//2)); sw=int(spec.get("scan_width",5))
        r=np.zeros((h,w,3),dtype=np.uint8)
        for y in range(h):
            for x in range(w):
                d=abs(x-sp)
                if d<sw: a3=1-d/sw; v=g[y,min(x,w-1)]; r[y,x]=(int(v*200*a3),int(v*220*a3),int(v*150*a3))
                else: v=g[y,min(x,w-1)]; r[y,x]=(int(v*40),int(v*50),int(v*30))
        for y in range(0,h,h//10): cv2.line(r,(0,y),(w,y),(60,80,50),1)
        arr=r
    elif effect == "sem":
        g=cv2.cvtColor(arr,cv2.COLOR_RGB2GRAY).astype(np.float32)/255
        gx=cv2.Sobel(g,cv2.CV_32F,1,0,ksize=3); gy=cv2.Sobel(g,cv2.CV_32F,0,1,ksize=3)
        m=np.sqrt(gx**2+gy**2); a4=float(spec.get("light_angle",45))
        sh=g+m*np.cos(math.radians(a4))*0.5; sh=sh/(sh.max()+1e-8)
        sv=(sh*230).clip(0,255).astype(np.uint8); r=np.stack([sv,sv,sv],axis=-1)
        bl=w//8; cv2.rectangle(r,(10,h-20),(10+bl,h-10),(255,255,255),-1)
        cv2.putText(r,"10um",(10,h-25),cv2.FONT_HERSHEY_SIMPLEX,0.4,(255,255,255),1)
        arr=r
    elif effect == "ct_scan":
        g=cv2.cvtColor(arr,cv2.COLOR_RGB2GRAY).astype(np.float32)/255; cn=float(spec.get("center",0.5)); wd=float(spec.get("width",0.3))
        wd2=np.clip((g-cn)/wd+0.5,0,1)
        if spec.get("positive",False): cv2=(wd2*255).clip(0,255).astype(np.uint8)
        else: cv2=((1-wd2)*255).clip(0,255).astype(np.uint8)
        r2=cv2.cvtColor(cv2.cvtColor(np.stack([cv2,cv2,cv2],axis=-1),cv2.COLOR_RGB2GRAY),cv2.COLOR_GRAY2RGB)
        r2[:,:,2]=(r2[:,:,2].astype(np.float32)*0.6).astype(np.uint8); r2[:,:,0]=(r2[:,:,0].astype(np.float32)*0.3).astype(np.uint8)
        arr=r2
    elif effect == "spotlight":
        cx2=int(spec.get("center_x",w//2)); cy2=int(spec.get("center_y",h//2)); r3=float(spec.get("radius",min(w,h)*0.3)); f2=float(spec.get("falloff",2))
        c3=spec.get("color",(255,240,200)); i2=float(spec.get("intensity",0.6))
        if isinstance(c3,str): c3=_parse_color(c3)
        yy,xx=np.mgrid[:h,:w].astype(np.float32); d=np.sqrt((xx-cx2)**2+(yy-cy2)**2)
        m=np.clip(1-(d/r3)**f2,0,1)
        dk=arr.astype(np.float32)*0.3; lt=arr.astype(np.float32)
        for c in range(3): lt[:,:,c]=np.clip(lt[:,:,c]*(1+(c3[c]/255-1)*i2),0,255)
        arr=(dk*(1-m[:,:,None])+lt*m[:,:,None]).clip(0,255).astype(np.uint8)
    elif effect == "gobo":
        p2=spec.get("pattern","grid"); sz=int(spec.get("size",30)); i3=float(spec.get("intensity",0.5)); c4=spec.get("color",(255,200,100))
        if isinstance(c4,str): c4=_parse_color(c4)
        yy,xx=np.mgrid[:h,:w]
        if p2=="grid": gb=((xx//sz)%2==0).astype(np.float32)*((yy//sz)%2==0).astype(np.float32)
        elif p2=="stripes": gb=(np.sin(xx/sz*3.14)>0).astype(np.float32)
        elif p2=="dots": cd2=xx%sz-sz//2; cd3=yy%sz-sz//2; gb=((cd2**2+cd3**2)<(sz//3)**2).astype(np.float32)
        elif p2=="radial": d2=np.sqrt((xx-w/2)**2+(yy-h/2)**2); gb=(np.sin(d2/sz)>0).astype(np.float32)
        else: gb=np.ones((h,w))
        gb=cv2.GaussianBlur(gb,(5,5),0)/gb.max()
        go=np.zeros((h,w,3),dtype=np.float32)
        for c in range(3): go[:,:,c]=gb*c4[c]
        arr=cv2.addWeighted(arr.astype(np.float32),1,go,i3,0).clip(0,255).astype(np.uint8)
    elif effect == "rim_light":
        a5=float(spec.get("angle",180)); wd3=float(spec.get("width",0.15)); i4=float(spec.get("intensity",0.5)); c5=spec.get("color",(255,220,180))
        if isinstance(c5,str): c5=_parse_color(c5)
        yy,xx=np.mgrid[:h,:w].astype(np.float32); lx=math.cos(math.radians(a5)); ly=math.sin(math.radians(a5))
        ed=xx*lx+yy*ly; ed=(ed-ed.min())/(ed.max()-ed.min()+1e-8)
        rim=np.exp(-((ed-1)/wd3)**2)*(1-np.exp(-ed/0.1)); rim=rim/rim.max()
        r4=arr.astype(np.float32)
        for c in range(3): r4[:,:,c]=r4[:,:,c]+rim*c5[c]*i4
        arr=r4.clip(0,255).astype(np.uint8)
    elif effect == "risograph":
        rc=spec.get("colors",[(200,80,50),(50,100,180),(50,180,80)]); ds=int(spec.get("dot_size",6)); random.seed(42)
        g=cv2.cvtColor(arr,cv2.COLOR_RGB2GRAY).astype(np.float32)/255; r=np.ones((h,w,3),dtype=np.uint8)*240
        for li,color in enumerate(rc):
            rx,ry=random.randint(-3,3)*li,random.randint(-3,3)*li
            for y in range(0,h,ds):
                for x in range(0,w,ds):
                    sx=np.clip(x+rx,0,w-1); sy=np.clip(y+ry,0,h-1)
                    avg=g[sy:min(sy+ds,h),sx:min(sx+ds,w)].mean()
                    if avg<0.7:
                        ad=1-avg
                        for dy in range(ds):
                            for dx2 in range(ds):
                                yy2,xx2=y+dy,x+dx
                                if yy2<h and xx2<w:
                                    for c in range(3): r[yy2,xx2,c]=int(r[yy2,xx2,c]*(1-ad*0.7)+color[c]*ad*0.7)
        arr=r
    elif effect == "gravure":
        cs=int(spec.get("cell_size",10)); g=cv2.cvtColor(arr,cv2.COLOR_RGB2GRAY).astype(np.float32)
        r=np.ones((h,w,3),dtype=np.uint8)*245
        for y in range(0,h,cs):
            for x in range(0,w,cs):
                avg=g[y:y+cs,x:x+cs].mean()/255
                cr2=max(1,int(cs*(1-avg)*0.8)); cx2=min(x+cs//2,w-1); cy2=min(y+cs//2,h-1)
                cv2.circle(r,(cx2,cy2),cr2,tuple(int(c)for c in arr[cy2,cx2]),-1)
        arr=r
    elif effect == "screen_print":
        ss=int(spec.get("screen",5)); g=cv2.cvtColor(arr,cv2.COLOR_RGB2GRAY).astype(np.float32)/255
        r=np.ones((h,w,3),dtype=np.uint8)*255
        cmyk_a=[15,75,0,45]; cmyk_c=[(100,180,255),(255,100,180),(255,255,100),(50,50,50)]
        for li in range(min(4,int(spec.get("layers",3)))):
            ar=math.radians(cmyk_a[li]); ca,sa=math.cos(ar),math.sin(ar)
            yy,xx=np.mgrid[:h,:w].astype(np.float32)
            sc2=(np.sin((xx*ca+yy*sa)/ss)>0).astype(np.float32)
            sa2=g*0.4*(1-li*0.15)
            for c in range(3):
                ch2=r[:,:,c].astype(np.float32)
                r[:,:,c]=(ch2*(1-sc2*sa2)+cmyk_c[li][c]*sc2*sa2).clip(0,255).astype(np.uint8)
        arr=r
    elif effect == "letterpress":
        g=cv2.cvtColor(arr,cv2.COLOR_RGB2GRAY).astype(np.float32)/255; pl2=1-g
        sh2=np.roll(pl2,2,0); sh2=np.roll(sh2,1,1)
        hl2=np.roll(pl2,-2,0); hl2=np.roll(hl2,-1,1)
        r=np.ones((h,w,3),dtype=np.uint8)*248; ink=np.clip(g,0,1)
        ic=spec.get("color",(30,20,15))
        if isinstance(ic,str): ic=_parse_color(ic)
        for c in range(3):
            ch2=r[:,:,c].astype(np.float32)
            ch2=ch2*(1-ink*0.9)+ic[c]*ink*0.9-sh2*15+hl2*10
            r[:,:,c]=ch2.clip(0,255).astype(np.uint8)
        arr=r
    elif effect == "protanopia":
        s=float(spec.get("severity",1)); rgb=arr.astype(np.float32)/255
        si=rgb.copy(); si[:,:,0]=rgb[:,:,0]*0+rgb[:,:,1]*0+rgb[:,:,2]*0; si[:,:,1]=rgb[:,:,1]*0.7+rgb[:,:,2]*0.3; si[:,:,2]=rgb[:,:,2]*0.8+rgb[:,:,1]*0.2
        arr=cv2.addWeighted(arr,1-s,(si*255).clip(0,255).astype(np.uint8),s,0)
    elif effect == "tritanopia":
        s=float(spec.get("severity",1)); rgb=arr.astype(np.float32)/255
        si=rgb.copy(); si[:,:,0]=rgb[:,:,0]*0.95; si[:,:,1]=rgb[:,:,0]*0.05+rgb[:,:,1]*0.95; si[:,:,2]=rgb[:,:,0]*0.05+rgb[:,:,2]*0.95
        arr=cv2.addWeighted(arr,1-s,(si*255).clip(0,255).astype(np.uint8),s,0)
    elif effect == "color_wash":
        c6=spec.get("color",(100,150,255)); s5=float(spec.get("strength",0.3))
        if isinstance(c6,str): c6=_parse_color(c6)
        g=cv2.cvtColor(arr,cv2.COLOR_RGB2GRAY).astype(np.float32)/255
        r=np.zeros_like(arr,dtype=np.float32)
        for c in range(3): r[:,:,c]=arr[:,:,c].astype(np.float32)*(1-g*s5*0.5)+c6[c]*g*s5*0.5
        arr=r.clip(0,255).astype(np.uint8)
    elif effect == "crt":
        r=arr.astype(np.float32)
        for i in range(0,h,2): r[i:i+1]*=0.7
        br=(cv2.cvtColor(arr,cv2.COLOR_RGB2GRAY)>200).astype(np.float32); br=cv2.GaussianBlur(br,(15,15),0)
        r+=br[:,:,None]*40
        cx3,cy3=w/2,h/2; yy,xx=np.mgrid[:h,:w].astype(np.float32); nx,ny=(xx-cx3)/cx3,(yy-cy3)/cy3
        r2=nx**2+ny**2; d3=1+0.08*r2
        r=cv2.remap(r,np.clip(cx3+nx*cx3/d3,0,w-1).astype(np.float32),np.clip(cy3+ny*cy3/d3,0,h-1).astype(np.float32),cv2.INTER_LINEAR)
        arr=r.clip(0,255).astype(np.uint8)
    elif effect == "psx":
        rs=float(spec.get("resolution",0.25)); sh2=int(spec.get("wobble_x",2)); sv2=int(spec.get("wobble_y",2))
        sm=max(2,int(h*rs)); sw2=max(2,int(w*rs))
        sm2=cv2.resize(arr,(sw2,sm),interpolation=cv2.INTER_NEAREST)
        if sh2 or sv2:
            sf=np.zeros_like(sm2); bh=max(4,sm//16); bw2=max(4,sw2//16)
            by=0
            while by<sm:
                bx=0
                while bx<sw2:
                    bw3=min(bw2,sw2-bx); bh3=min(bh,sm-by)
                    sx=np.clip(bx+random.randint(-sh2,sh2),0,sw2-bw3); sy=np.clip(by+random.randint(-sv2,sv2),0,sm-bh3)
                    sf[by:by+bh3,bx:bx+bw3]=sm2[sy:sy+bh3,sx:sx+bw3]
                    bx+=bw2
                by+=bh
            sm2=sf
        sm2=(sm2.astype(np.float32)//8*8+4).clip(0,255).astype(np.uint8)
        arr=cv2.resize(sm2,(w,h),interpolation=cv2.INTER_NEAREST)
        if spec.get("dither",True):
            yy,xx=np.mgrid[:h,:w]; dt=((xx^yy)&4).astype(np.uint8)*8
            arr=(arr.astype(np.float32)+dt[:,:,None]*0.3).clip(0,255).astype(np.uint8)
    elif effect == "ntsc":
        r,g,b=arr.astype(np.float32).transpose(2,0,1)
        y2=0.299*r+0.587*g+0.114*b; i5=0.596*r-0.274*g-0.322*b; q5=0.211*r-0.523*g+0.312*b
        fq=float(spec.get("frequency",0.5)); cr5=np.sin(np.arange(w)*fq)
        i5+=cr5*20; q5+=cr5*15
        yy,xx=np.mgrid[:h,:w]; dc=np.sin(xx*0.3+yy*0.2)>0
        r2=y2+0.956*i5+0.621*q5; g2=y2-0.272*i5-0.647*q5; b2=y2-1.106*i5+1.703*q5
        an=np.stack([r2,g2,b2],axis=-1); an[dc]=an[dc]*0.85+25
        gh2=np.roll(an,3,axis=1); gh2[:,:3]=an[:,:3]; an=an*0.7+gh2*0.3
        arr=an.clip(0,255).astype(np.uint8)
    elif effect == "stretch":
        ax=spec.get("axis","x"); a=float(spec.get("amount",0.5))
        yy,xx=np.mgrid[:h,:w].astype(np.float32)
        if ax=="x":
            cx4=w/2; sf2=1+a*np.abs(xx-cx4)/cx4
            arr=cv2.remap(arr,np.clip(cx4+(xx-cx4)/sf2,0,w-1).astype(np.float32),yy.astype(np.float32),cv2.INTER_LINEAR)
        else:
            cy4=h/2; sf2=1+a*np.abs(yy-cy4)/cy4
            arr=cv2.remap(arr,xx.astype(np.float32),np.clip(cy4+(yy-cy4)/sf2,0,h-1).astype(np.float32),cv2.INTER_LINEAR)
    elif effect == "accordion":
        f=int(spec.get("folds",8)); a=float(spec.get("amplitude",30)); ax=spec.get("axis","x")
        yy,xx=np.mgrid[:h,:w].astype(np.float32)
        if ax=="x": off=a*np.sin(xx*f/w*math.pi); arr=cv2.remap(arr,np.clip(xx+off,0,w-1).astype(np.float32),yy.astype(np.float32),cv2.INTER_LINEAR)
        else: off=a*np.sin(yy*f/h*math.pi); arr=cv2.remap(arr,xx.astype(np.float32),np.clip(yy+off,0,h-1).astype(np.float32),cv2.INTER_LINEAR)
    elif effect == "explosion":
        s=float(spec.get("strength",0.4)); cx5,cy5=w/2,h/2
        yy,xx=np.mgrid[:h,:w].astype(np.float32); dx,dy=xx-cx5,yy-cy5; d=np.sqrt(dx**2+dy**2)/np.sqrt(cx5**2+cy5**2)
        p=s*np.exp(-d*2)
        arr=cv2.remap(arr,np.clip(cx5+dx*(1+p),0,w-1).astype(np.float32),np.clip(cy5+dy*(1+p),0,h-1).astype(np.float32),cv2.INTER_LINEAR)
    elif effect == "implosion":
        s=float(spec.get("strength",0.3)); cx6,cy6=w/2,h/2
        yy,xx=np.mgrid[:h,:w].astype(np.float32); dx,dy=xx-cx6,yy-cy6; d=np.sqrt(dx**2+dy**2)/np.sqrt(cx6**2+cy6**2)
        p=s*np.exp(-d*2)
        arr=cv2.remap(arr,np.clip(cx6+dx*(1-p),0,w-1).astype(np.float32),np.clip(cy6+dy*(1-p),0,h-1).astype(np.float32),cv2.INTER_LINEAR)
    elif effect == "smear":
        a=float(spec.get("angle",0)); d=int(spec.get("distance",40)); i=float(spec.get("intensity",0.5))
        rad=math.radians(a); dx=int(d*math.cos(rad)); dy=int(d*math.sin(rad))
        r5=np.zeros_like(arr,dtype=np.float32); ct5=np.zeros((h,w),dtype=np.float32)
        steps=max(1,int(i*20))
        for s in range(steps):
            ox,oy=int(dx*s/steps),int(dy*s/steps)
            sh5=np.roll(arr,oy,0); sh5=np.roll(sh5,ox,1)
            if ox>0: sh5[:,:ox]=0
            elif ox<0: sh5[:,ox:]=0
            if oy>0: sh5[:oy]=0
            elif oy<0: sh5[oy:]=0
            r5+=sh5.astype(np.float32); ct5+=(sh5>0).any(axis=2).astype(np.float32)
        ct5=np.maximum(ct5,1); r5=r5/ct5[:,:,None]; m2=ct5>1
        for c in range(3): r5[~m2,c]=arr[~m2,c].astype(np.float32)
        arr=r5.clip(0,255).astype(np.uint8)
    elif effect == "film_scratches":
        ns=int(spec.get("scratches",15)); nd=int(spec.get("dust",200)); random.seed(42)
        r=arr.copy().astype(np.float32)
        for _ in range(ns):
            x=random.randint(0,w-1); y=random.randint(0,h-1); l=random.randint(30,h-y)
            cv2.line(r,(x,y),(x,y+l),(255,255,255)if random.random()<0.5 else(0,0,0),random.randint(1,2))
        for _ in range(nd):
            cv2.circle(r,(random.randint(0,w-1),random.randint(0,h-1)),random.randint(1,4),(30,30,30),-1)
        arr=r.clip(0,255).astype(np.uint8)
    elif effect == "film_gate":
        a=float(spec.get("amount",0.3))
        r=np.zeros_like(arr)
        for y in range(h):
            j=int(a*5*(np.sin(y*0.1)*0.5+0.5)); r[y]=np.roll(arr[y],j,axis=0)
        vw=a*3*np.sin(np.arange(h)*0.05)
        for y in range(h):
            if vw[y]>1: r[y]=cv2.addWeighted(r[y],0.6,r[max(0,min(h-1,y-1))],0.4,0)
        arr=r
    elif effect == "hex_grid":
        hs=int(spec.get("size",12)); dx2=int(hs*1.5); dy2=int(hs*np.sqrt(3))
        r=np.zeros((h,w,3),dtype=np.uint8)
        for y in range(-dy2,h+dy2,dy2):
            for x in range(-dx2,w+dx2,dx2):
                ox6=(y//dy2)%2*dx2//2; cx7=x+ox6+dx2//2; cy7=y+dy2//2
                sx=np.clip(cx7,0,w-1); sy=np.clip(cy7,0,h-1); c=tuple(int(ci)for ci in arr[sy,sx])
                pts=np.array([(int(cx7+hs*math.cos(math.pi/3*a-math.pi/6)),int(cy7+hs*math.sin(math.pi/3*a-math.pi/6)))for a in range(6)],dtype=np.int32)
                cv2.fillPoly(r,[pts],c); cv2.polylines(r,[pts],True,(0,0,0),max(1,hs//12))
        arr=r
    elif effect == "isometric":
        ts=int(spec.get("tile_size",20)); tw=ts*2; th=ts
        r=np.zeros((h,w,3),dtype=np.uint8)
        for y in range(-ts,h+th,th):
            for x in range(-tw,w+tw,tw):
                ox7=(y//th)%2*ts; cx8=x+ox7+tw//2; cy8=y+th//2
                sx=np.clip(cx8,0,w-1); sy=np.clip(cy8,0,h-1); c=tuple(int(ci)for ci in arr[sy,sx])
                pts=np.array([(cx8,cy8-ts),(cx8+ts,cy8),(cx8,cy8+ts),(cx8-ts,cy8)],dtype=np.int32)
                cv2.fillPoly(r,[pts],c); cv2.polylines(r,[pts],True,(0,0,0),1)
        arr=r
    elif effect == "tile_mirror":
        tx=int(spec.get("tiles_x",4)); ty=int(spec.get("tiles_y",4))
        tw=w//tx; th=h//ty; r=np.zeros((h,w,3),dtype=np.uint8)
        for ty2 in range(ty):
            for tx2 in range(tx):
                x0,x1=tx2*tw,min((tx2+1)*tw,w); y0,y1=ty2*th,min((ty2+1)*th,h)
                src=cv2.resize(arr,(x1-x0,y1-y0),interpolation=cv2.INTER_LINEAR)
                if tx2%2==1: src=src[:,::-1]
                if ty2%2==1: src=src[::-1,:]
                r[y0:y1,x0:x1]=src
        gr=int(spec.get("grout",2))
        for y2 in range(0,h,th): r[y2:min(y2+gr,h)]=(200,200,200)
        for x2 in range(0,w,tw): r[:,x2:min(x2+gr,w)]=(200,200,200)
        arr=r
    elif effect == "underwater":
        d=float(spec.get("depth",0.5))
        r6=arr[:,:,0].astype(np.float32)*(0.5-d*0.3); g6=arr[:,:,1].astype(np.float32)*(0.7+d*0.2); b6=arr[:,:,2].astype(np.float32)*(0.9+d*0.4)
        yy,xx=np.mgrid[:h,:w].astype(np.float32)
        ca7=np.sin(xx*0.05+yy*0.03)*np.cos(xx*0.03-yy*0.05)
        ca7=(ca7+1)/2*0.3
        au=np.stack([r6,g6,b6],axis=-1)*(1+ca7[:,:,None])
        bu=max(1,int(d*6)); bu+=1 if bu%2==0 else 0
        au=cv2.GaussianBlur(au,(bu,bu),0)
        random.seed(42)
        for _ in range(int(10*d)):
            x=random.randint(0,w); cv2.line(au,(x,0),(x+random.randint(-20,20),h),(100,150,200),random.randint(1,3))
        arr=au.clip(0,255).astype(np.uint8)
    elif effect == "snowfall":
        d=int(spec.get("density",300)); random.seed(42)
        o=np.zeros((h,w,3),dtype=np.float32)
        for _ in range(d):
            x=random.randint(0,w-1); y=random.randint(0,h-1); r7=random.randint(1,3); tr=random.randint(3,8)
            for t in range(tr):
                yy2=min(y-t*2,h-1); xx2=np.clip(x+t*random.randint(-1,1),0,w-1)
                cv2.circle(o,(xx2,yy2),max(1,r7-t//2),(255,255,255),-1)
        o=cv2.GaussianBlur(o,(3,3),0)
        arr=cv2.addWeighted(arr.astype(np.float32),1,o,0.7,0).clip(0,255).astype(np.uint8)
    elif effect == "sandstorm":
        i=float(spec.get("intensity",0.5)); sc7=np.array([180,150,100],dtype=np.float32)
        a=arr.astype(np.float32)*(1-i*0.5)+sc7[None,None,:]*i*0.5
        yy,xx=np.mgrid[:h,:w].astype(np.float32)
        gr7=np.sin(xx*0.08+yy*0.01+np.random.randn(h,w)*2)
        gr7=((gr7+1)/2*i*40).clip(0,255).astype(np.uint8)
        for c in range(3): a[:,:,c]=(a[:,:,c]+gr7*0.3).clip(0,255)
        gs=cv2.cvtColor(a.clip(0,255).astype(np.uint8),cv2.COLOR_RGB2GRAY).astype(np.float32)
        gs=gs*(1-i*0.3)+128*i*0.3
        arr=cv2.addWeighted(a.clip(0,255).astype(np.uint8),0.7,cv2.cvtColor(gs.clip(0,255).astype(np.uint8),cv2.COLOR_GRAY2RGB),0.3,0)
    elif effect == "blend_with":
        bp=spec.get("path",""); bm=spec.get("mode","overlay"); ab=float(spec.get("alpha",0.5))
        if bp:
            try:
                o=cv2.imread(bp)
                if o is not None:
                    o=cv2.resize(cv2.cvtColor(o,cv2.COLOR_BGR2RGB),(w,h),interpolation=cv2.INTER_LINEAR).astype(np.float32)
                    af=arr.astype(np.float32)
                    if bm=="overlay": r8=np.where(af<128,2*af*o/255,255-2*(255-af)*(255-o)/255)
                    elif bm=="screen": r8=255-(255-af)*(255-o)/255
                    elif bm=="multiply": r8=af*o/255
                    elif bm=="difference": r8=np.abs(af-o)
                    else: r8=af*(1-ab)+o*ab
                    arr=r8.clip(0,255).astype(np.uint8)
            except: pass
    elif effect == "picture_in_picture":
        sc=float(spec.get("scale",0.25)); c9=spec.get("corner","br")
        ph2, pw2=int(h*sc),int(w*sc)
        sm=cv2.resize(arr,(pw2,ph2),interpolation=cv2.INTER_LINEAR)
        b9=4; pw=np.ones((ph2+b9*2,pw2+b9*2,3),dtype=np.uint8)*255
        pw[b9:b9+ph2,b9:b9+pw2]=sm
        if c9=="br": y0,x0=h-ph2-b9*2-10,w-pw2-b9*2-10
        elif c9=="bl": y0,x0=h-ph2-b9*2-10,10
        elif c9=="tr": y0,x0=10,w-pw2-b9*2-10
        else: y0,x0=10,10
        y0,x0=max(0,y0),max(0,x0); y1=min(y0+ph2+b9*2,h); x1=min(x0+pw2+b9*2,w)
        arr[y0:y1,x0:x1]=pw[0:y1-y0,0:x1-x0]
    elif effect == "moire":
        sc=float(spec.get("scale",0.04)); i=float(spec.get("intensity",0.5))
        yy,xx=np.mgrid[:h,:w].astype(np.float32)
        g1=np.sin(xx*sc*10+yy*sc*3); g2=np.sin(xx*sc*9+yy*sc*5); p=np.sin(g1*g2*3); p=(p+1)/2
        mo=np.stack([((np.sin(p*6.28)+1)*0.5*255).clip(0,255).astype(np.uint8),((np.sin(p*6.28+2.09)+1)*0.5*255).clip(0,255).astype(np.uint8),((np.sin(p*6.28+4.19)+1)*0.5*255).clip(0,255).astype(np.uint8)],axis=-1)
        arr=cv2.addWeighted(arr,1-i,mo,i,0)
    elif effect == "afterimage":
        s=int(spec.get("shift",20)); a=float(spec.get("alpha",0.4)); n=int(spec.get("copies",4))
        r=np.zeros_like(arr,dtype=np.float32)
        for i in range(n):
            s2=int(s*(i+1)/n); c2=np.roll(arr.astype(np.float32),s2,1); c2=np.roll(c2,s2//2,0)
            if i%2==1: c2=255-c2
            r+=c2*a/n*(1-i/n)
        arr=cv2.addWeighted(arr.astype(np.float32),0.5,r,1,0).clip(0,255).astype(np.uint8)
    elif effect == "checkerboard":
        sc=int(spec.get("scale",20)); d=float(spec.get("depth",1))
        yy,xx=np.mgrid[:h,:w]; ch=((xx//sc)+(yy//sc))%2; r=np.zeros_like(arr)
        for c in range(3):
            ch2=arr[:,:,c].astype(np.float32)
            r[:,:,c]=np.where(ch==1,ch2*(0.5+d*0.5),ch2*(1-d*0.5)).clip(0,255).astype(np.uint8)
        arr=r
    elif effect == "denim":
        sc=int(spec.get("scale",4)); r=np.zeros((h,w,3),dtype=np.float32)
        for y in range(h):
            for x in range(w):
                cy=min(y//sc,h-1); cx=min(x//sc,w-1); b=arr[cy,cx].astype(np.float32)
                tw=((x//sc)+(y//sc))%4; tv=(tw-1.5)/3*30; yn=np.sin(y*0.5+x*0.3)*8
                r[y,x]=[b[0]*0.85+tv*0.5+yn*0.3+10,b[1]*0.7+tv*0.3-5,b[2]*1.1+tv*0.4+15]
        arr=r.clip(0,255).astype(np.uint8)
    elif effect == "brushed_metal":
        sc=float(spec.get("scale",0.05)); i=float(spec.get("intensity",0.4))
        g=cv2.cvtColor(arr,cv2.COLOR_RGB2GRAY).astype(np.float32)/255
        yy,xx=np.mgrid[:h,:w].astype(np.float32)
        br=np.sin(xx*sc*50+yy*sc*2+np.random.randn(h,w)*0.5); br=(br+1)/2
        m=g*0.6+br*0.4; r=np.stack([m*200,m*200,m*220],axis=-1).clip(0,255).astype(np.uint8)
        arr=cv2.addWeighted(arr,1-i,r,i,0)
    elif effect == "wood_grain":
        sc=float(spec.get("scale",0.02)); i=float(spec.get("intensity",0.5))
        g=cv2.cvtColor(arr,cv2.COLOR_RGB2GRAY).astype(np.float32)/255; yy,xx=np.mgrid[:h,:w].astype(np.float32)
        dc=np.sqrt((xx-w*0.7)**2+(yy-h*0.5)**2)
        gr2=np.sin(dc*sc*40)*0.5+0.5; gr2=(gr2*(1+np.random.randn(h,w)*0.1)).clip(0,1)
        wd=np.stack([(g*150+gr2*80).clip(0,255).astype(np.uint8),(g*100+gr2*40).clip(0,255).astype(np.uint8),(g*50+gr2*20).clip(0,255).astype(np.uint8)],axis=-1)
        arr=cv2.addWeighted(arr,1-i,wd,i,0)
    elif effect == "marble":
        sc=float(spec.get("scale",0.015)); i=float(spec.get("intensity",0.6))
        yy,xx=np.mgrid[:h,:w].astype(np.float32)
        mp=np.sin(xx*sc*6+yy*sc*4)*0.5+np.sin(xx*sc*12+yy*sc*9)*0.25+np.sin(xx*sc*25+yy*sc*18)*0.125; mp=(mp+1)/2
        vn=np.sin(mp*20)**2; b=np.ones((h,w,3),dtype=np.uint8)*240
        for c in range(3): b[:,:,c]=(b[:,:,c].astype(np.float32)-vn*80*(1-c*0.3)).clip(0,255).astype(np.uint8)
        arr=cv2.addWeighted(arr,1-i,b,i,0)
    elif effect == "leather":
        sc=float(spec.get("scale",0.1)); i=float(spec.get("intensity",0.4))
        g=cv2.cvtColor(arr,cv2.COLOR_RGB2GRAY).astype(np.float32)/255; yy,xx=np.mgrid[:h,:w].astype(np.float32)
        gr2=np.sin(xx*sc*15+yy*sc*10)*0.3+np.sin(xx*sc*30+yy*sc*22)*0.2+np.sin(xx*sc*50+yy*sc*40)*0.1; gr2=(gr2+0.6).clip(0,1)
        le=np.stack([(g*160+gr2*60).clip(0,255).astype(np.uint8),(g*100+gr2*40).clip(0,255).astype(np.uint8),(g*50+gr2*20).clip(0,255).astype(np.uint8)],axis=-1)
        arr=cv2.addWeighted(arr,1-i,le,i,0)

    elif effect == "shader":
        from .shaders import render_filter as _render_shader_filter, list_shaders as _list_shaders
        shader_name = spec.get("shader", "shader_bloom")
        params = (float(spec.get("p1", 0.5)), float(spec.get("p2", 0.5)),
                  float(spec.get("p3", 0.5)), float(spec.get("p4", 0.5)))
        time = float(spec.get("time", 0.0))
        result_img = _render_shader_filter(shader_name, arr, params, time)
        arr = np.array(result_img, dtype=np.uint8)

    else:
        print(f"  ⚠ Unknown filter effect: '{effect}'")


    # Save result
    result_img = Image.fromarray(arr)
    result_img.save(str(out_path))
    return out_path


def apply_filter_batch(out_dir, method_ids, filter_spec, suffix=""):
    """Apply a filter to all generated images for the given method IDs."""
    from .registry import get_meta
    results = []
    for mid in method_ids:
        meta = get_meta(mid)
        if not meta:
            continue
        src = out_dir / meta.filename()
        if not src.exists():
            continue
        if suffix:
            dst_name = meta.filename().replace(".png", f"-{suffix}.png")
            dst = out_dir / dst_name
        else:
            dst = src
        apply_filter(src, filter_spec, dst)
        results.append(dst)
    return results
