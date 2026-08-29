import math
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

def multyp(a,b):
    #("multiplication starts"
    #("dimensions of matrix A")
    ra,ca=len(a),len(a[0])
    #("matrix b dimensions")
    rb=len(b)
    cb=len(b[0])
    #check if column of A=row of B
    if ca!=rb:
        print("multiplication is not possible")
        return
    #"mutliplication of a and b if column(a)=row(b)"
    ar=[]
    for m in range(ra):
        rtw=[]
        for n in range(cb):
            tola=0
            for o in range(ca):

                tola=tola+a[m][o]*b[o][n]
                er=tola
            rtw.append(tola)
        ar.append(rtw)
    #multiplication ends
    return ar

def load_grayscale_image(image_path):
#black # white
    img=Image.open(image_path).convert("L")
    w,h=img.size
    if w==h:
        img=img.resize((98,98))
    else:
        img=img.resize((80,98))
    A=[]
    for i in range(img.height):
        row=[]
        for j in range(img.width):
            row.append(float(img.getpixel((j,i))))
        A.append(row)
    return A


def tranfese(a):
    rs=len(a)
    cs=len(a[0])
    result=[]
    for j in range(cs):
        rw=[]
        for i in range(rs):
            rw.append(a[i][j])
        result.append(rw)
    return result

def diagl(v):
    ny=len(v)#find length
    dq=[]
    for i in range(ny):
        rw=[]
        for j in range(ny):
            if i==j:
                rw.append(v[i])
            else:
                rw.append(0)
                #find diagonal matrix
        dq.append(rw)
    return dq

def jacobi(a):
    # to fInd the size of mAtrix
    n=len(a)

    # identity matrix \\\eigenvectors\\
    ve=[]
    for i in range(n):
        rw=[]
        for j in range(n):
            rw.append(1.0 if i==j else 0.0)
        ve.append(rw)

    #do << Jacobi rotation sweeps>
    for sweep in range(100):
        maximum=0

        #largest_off-diagonal element//
        for w1 in range(n-1):
            for w2 in range(w1+1,n):
                maximum=max(maximum,abs(a[w1][w2]))

        #condition:true? stops
        #nearly diagonalmatrix
        if maximum<1e-10:
            break

        #find rotations  off-diagonal elements
        for w1 in range(n-1):
            for w2 in range(w1+1,n):
                if abs(a[w1][w2])<1e-12:
                    continue

                #applurotation angle
                theta=0.5*math.atan2(2*a[w1][w2],a[w2][w2]-a[w1][w1])
                c=math.cos(theta)
                s=math.sin(theta)

                #update values of matrix
                for i in range(n):
                    o=i
                    if i!=w1 and o!=w2:
                        aip=a[i][w1]
                        aiq=a[o][w2]
                        a[i][w1]=c*aip-s*aiq
                        a[w1][i]=a[i][w1]
                        a[o][w2]=s*aip+c*aiq
                        a[w2][i]=a[o][w2]

                #store values required for calcluation
                f=w1
                g=w2
                rit=a[w1][f]
                rit2=a[g][w2]
                apq=a[w1][w2]

                # Update diagonal elements
                a[f][w1]=c*c*rit-2*s*c*apq+s*s*rit2
                a[g][w2]=s*s*rit+2*s*c*apq+c*c*rit2

                #  off-diagonal 'elements==zero
                a[f][g]=0
                a[w2][w1]=0

                #in this update the eigenvalue
                for i in range(n):
                    vip=ve[i][w1]
                    viq=ve[i][w2]
                    ve[i][w1]=c*vip-s*viq
                    ve[i][w2]=s*vip+c*viq

    #extracting eigenvalues
    #it is done from diagonals
    eigenvalues=[]
    for i in range(n):
        eigenvalues.append(a[i][i])
#discovered all necessary values like eigenvalue and eigenvectors..
    return eigenvalues,ve

def sortevs(eigenvalues,eigenvectors):
    n=len(eigenvalues)
    pairs=[]
    for i in range(n):
        vector=[]
        for j in range(n):
            vector.append(eigenvectors[j][i])
        pairs.append((eigenvalues[i],vector))
    for i in range(n):
        max_index=i
        for j in range(i+1,n):
            if pairs[j][0]>pairs[max_index][0]:
                max_index=j
        pairs[i],pairs[max_index]=pairs[max_index],pairs[i]
    values=[]
    vectors=[]
    for i in range(n):
        values.append(pairs[i][0])
        vectors.append(pairs[i][1])
    result=[]
    for i in range(n):
        row=[]
        for j in range(n):
            row.append(vectors[j][i])
        result.append(row)
    return values,result

def calculate_svd(a):
    m=len(a)
    n=len(a[0])
    at=tranfese(a)
    ata=multyp(at,a)
    eigenvalues,v=jacobi(ata)
    eigenvalues,v=sortevs(eigenvalues,v)
    for i in range(len(eigenvalues)):
        if eigenvalues[i]<0:
            eigenvalues[i]=0
    s=[]
    for i in range(len(eigenvalues)):
        s.append(math.sqrt(eigenvalues[i]))
    r=min(m,n)
    s1=[]
    for i in range(r):
        s1.append(s[i])
    v1=[]
    for i in range(n):
        row=[]
        for j in range(r):
            row.append(v[i][j])
        v1.append(row)
    u=[]
    for i in range(m):
        row=[]
        for j in range(r):
            row.append(0.0)
        u.append(row)
    for i in range(r):
        if s1[i]>1e-12:
            for j in range(m):
                tola=0
                for k in range(n):
                    tola+=a[j][k]*v1[k][i]
                u[j][i]=tola/s1[i]
    return u,s1,tranfese(v1)

def svdreconstruction(a,k):
    u,s,v=calculate_svd(a)
    k=min(k,len(s))
    uk=[]
    for i in range(len(u)):
        row=[]
        for j in range(k):
            row.append(u[i][j])
        uk.append(row)
    sk=diagl(s[:k])
    vk=[]
    for i in range(k):
        row=[]
        for j in range(len(v[0])):
            row.append(v[i][j])
        vk.append(row)
    return multyp(multyp(uk,sk),vk)

def clean_matrix(A):
    result=[]
    for i in range(len(A)):
        row=[]
        for j in range(len(A[0])):
            value=A[i][j]
            value=max(0,min(255,value))
            row.append(value)
        result.append(row)
    return result

def froerror(aw,se):
    #assume total=0
    lalo=0
    #find the forbenius error
    for x in range(len(aw)):
        for y in range(len(aw[0])):
          #find the difference
            df=aw[x][y]-se[x][y]
            ra=df
            lalo+=ra*df
    return math.sqrt(lalo)

def error_image(At,Rt):
    res=[]
    for i in range(len(At)):
        row=[]
        for j in range(len(At[0])):
            row.append(abs(At[i][j]-Rt[i][j]))
        res.append(row)
    return res

def process_image(image_path):
    A=load_grayscale_image(image_path)
    print("\nImage:",image_path)
    print("Image converted to grayscale")
    print("Matrix dimension:",len(A),"x",len(A[0]))
    u,s,v=calculate_svd(A)
    print("\nSingular values:")
    for value in s:
        print(round(value,3))
    r=len(s)
    errors=[]
    for k in range(1,r+1):
        R=svdreconstruction(A,k)
        R=clean_matrix(R)
        error=froerror(A,R)
        errors.append(error)
        print("k =",k,"Error value=",round(error,3))
    k1=r%15
    k2=r>>2
    k3=r//2
    k_val=[k1,k2,k3]
    print("\nSVD images")
    for k in k_val:
        R=svdreconstruction(A,k)
        R=clean_matrix(R)
        error=froerror(A,R)
        error_img=error_image(A,R)
        plt.figure(figsize=(12,4))
        plt.subplot(1,3,1)
        plt.imshow(A,cmap="gray",vmin=0,vmax=255)
        plt.title("Original",color="red")
        plt.axis("off")
        plt.subplot(1,3,2)
        plt.imshow(R,cmap="gray",vmin=0,vmax=255)
        plt.title("SVD, k="+str(k),color="green")
        plt.axis("off")
        plt.subplot(1,3,3)
        plt.imshow(error_img,cmap="gray")
        plt.title("SVD Error\nFrobenius Error = "+str(round(error,3)),color="blue")
        plt.axis("off")
        plt.tight_layout()
        plt.show()
    plt.figure(figsize=(9,6))
    plt.plot(range(1,r+1),errors,marker="*",markersize=2,color="orange",label="SVD")
    font={'color':'green','size':16}
    font2={'color':'blue','size':20}
    plt.xlabel("Retained Component (k)",font)
    plt.ylabel("Frobenius Reconstruction Error",font)
    plt.title("Frobenius Reconstruction Error vs Retained Component - "+image_path,font2)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

process_image("cat11.png")

process_image("rectcat.png")
