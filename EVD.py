import math
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np


def multyp(sa,db):
 #multiplication starts
   #*dimensions of matrix sa'
    ra,ca=len(sa),len(sa[0])
 #matrix rb dimensions
    rb=len(db)#row of matrix2
    cb=len(db[0])
    #check if column1=row2
    if ca!=rb:# checking process ?
  #:NO multiplication
        print("multiplication is not possible")
        return
    #"multiplication_can be done"
    wri=[]
    for m in range(ra):
        rw=[]
        for n in range(cb):
            tola=0
            for o in range(ca):
              #stores multiplication final result]
                tola=tola+sa[m][o]*db[o][n]
            rw.append(tola)
        wri.append(rw)
    #multiplication ends/
    return wri


def tranfese(vo):
    #find length
    ol=len(vo)
    #square row=colum
    t1=ol
    t2=ol
    #transpose calculation .. starts
    aei=[]
    for x in range(t1):
        t=[]
        for y in range(t2):
            t.append(vo[y][x])
        aei.append(t)
    #end [transpose/]
    #completed
    return aei

def inverse(a):
    n=len(a)
    aug=[]
    for i in range(n):
        row=[]
        for j in range(n):
            row.append(complex(a[i][j]))
            #create identity ''0 when i!=j'
        #1 when i==j
        for j in range(n):
            if i==j:
                row.append(1)
            else:
                row.append(0)
        aug.append(row)
        #apply gaussian elimination
    for i in range(n):
        pivot=i
        for j in range(i+1,n):
            if abs(aug[j][i])>abs(aug[pivot][i]):
                pivot=j
        if abs(aug[pivot][i])<1e-10:
            print("not invertible")#To Check if matrix is invertable
            return
            #swapsrows
        temp=aug[i]
        aug[i]=aug[pivot]
        aug[pivot]=temp
        p=aug[i][i]#make pivot_vlue=1}
        for j in range(2*n):
            aug[i][j]=aug[i][j]/p
        for j in range(n):
            if j==i:
                continue
            factor=aug[j][i]
            for k in range(2*n):
                aug[j][k]=aug[j][k]-factor*aug[i][k]
                #extract the inverse matrix
    rt=[]
    for i in range(n):
        rw=[]
        for j in range(n):
            rw.append(aug[i][j+n])
        rt.append(rw)
        #return '' inverse matrix'''
    return rt


def diagonal(vr):
  #length matrix>
    nt=len(vr)
    #calculation:starts'
    dr=[]
    for g in range(nt):
        rw=[]
        for h in range(nt):
            if g==h:
                rw.append(vr[g])
            else:
                rw.append(0)
        dr.append(rw)
        #calculation stops /
        
    return dr


def sortevs(evf,feva):
    n=len(evf)
    tola=[]
    for i in range(n):
        vectorstodo=[]
        for j in range(n):
            vectorstodo.append(feva[j][i])
        tola.append((evf[i],vectorstodo))


    for i in range(n):
        tomaix=i
        for j in range(i+1,n):
            if abs(tola[j][0])>abs(tola[tomaix][0]):
                tomaix=j
        tola[i],tola[tomaix]=tola[tomaix],tola[i]


    vau=[]
    vectors=[]
    for i in range(n):
        vau.append(tola[i][0])
        vectors.append(tola[i][1])


    tola=[]
    for i in range(n):
        row=[]
        for j in range(n):
            row.append(vectors[j][i])
        tola.append(row)


    return vau,tola


def evdreconstruction(A,eigenvalues,eigenvectors,k):
    n=len(eigenvalues)
    eigenvalues,eigenvectors=sortevs(eigenvalues,eigenvectors)
    used=[False]*n
    components=[]
    for i in range(n):
        if used[i]:
            continue
        value=eigenvalues[i]
        if abs(value.imag)>1e-10:
            found=False
            for j in range(n):
                if i==j or used[j]:
                    continue
                if abs(eigenvalues[j]-value.conjugate())<1e-10:
                    components.append({"indices":[i,j],"value":abs(value)})
                    used[i]=True
                    used[j]=True
                    found=True
                    break
            if not found:
                components.append({"indices":[i],"value":abs(value)})
                used[i]=True
        else:
            components.append({"indices":[i],"value":abs(value)})
            used[i]=True
    for i in range(len(components)):
        for j in range(i+1,len(components)):
            if components[j]["value"]>components[i]["value"]:
                components[i],components[j]=components[j],components[i]
    selected=[]
    count=0
    for component in components:
        size=len(component["indices"])
        if count+size>k:
            continue
        for index in component["indices"]:
            selected.append(index)
        count=count+size
    lambda_k=[]
    for i in range(n):
        if i in selected:
            lambda_k.append(eigenvalues[i])
        else:
            lambda_k.append(0)
    Lambda=diagonal(lambda_k)
    Q=eigenvectors
    Qinv=inverse(Q)
    result=multyp(Q,Lambda)
    result=multyp(result,Qinv)
    for i in range(len(result)):
        for j in range(len(result[0])):
            if abs(result[i][j].imag)<1e-10:
                result[i][j]=result[i][j].real
    return result


def eagim(jy):
    img=Image.open(jy).convert("L").resize((98,98))
#convert.. img to array[__]/
    ry=[]
    for ti in range(img.height):
        tv=[]
        for to in range(img.width):
            tv.append(float(img.getpixel((to,ti))))
        ry.append(tv)
#completed : -
    return ry


def convert_to_list(A):
    qa=len(A)
    cols=len(A[0])
    matx=[]
    for i in range(qa):
        row=[]
        for j in range(cols):
            row.append(complex(A[i][j]))
        matx.append(row)
    return matx


def clean_matrix(A):
    result=[]
    for i in range(len(A)):
        row=[]
        for j in range(len(A[0])):
            value=A[i][j]
            if isinstance(value,complex):
                if abs(value.imag)<1e-8:
                    value=value.real
                else:
                    value=abs(value)
            value=max(0,min(255,value))
            row.append(value)
        result.append(row)
    return np.array(result,dtype=float)


def froerror(aw,aq):
    #assume total=0
    lalo=0
    #find the Frobenius error
    for x in range(len(aw)):
        for y in range(len(aw[0])):
          #find the difference
            df=aw[x][y]-aq[x][y]
            ra=df
            lalo+=ra*df
    return math.sqrt(lalo)
#the image
jy="cat11.png"
Alo=eagim(jy)
#find img size
A_lst=convert_to_list(Alo)
#To find:[[eigenvalues and eigenvectors]]
eigenvalues,eigenvectors=np.linalg.eig(Alo)
eigenvalues=[complex(x) for x in eigenvalues]
eigenvectors=[[complex(eigenvectors[i][j]) for j in range(len(eigenvectors[i]))] for i in range(len(eigenvectors))]
print("\nEigenvalues:")
for value in eigenvalues:
    print(value)


r=len(eigenvalues)
k_values=list(range(1,r+1))
errors=[]


for k in k_values:
    R=evdreconstruction(Alo,eigenvalues,eigenvectors,k)
    R=clean_matrix(R)
    error=froerror(Alo,R)
    errors.append(error)
    print("k =",k,"Error value =",round(error,3))


k_val=[5,25,r//2]
#..EVD images
for k in k_val:
    Rq=evdreconstruction(Alo,eigenvalues,eigenvectors,k)
    Rq=clean_matrix(Rq)
    error=froerror(Alo,Rq)
    error_image=abs(Alo-Rq)
    plt.figure(figsize=(12,4))
    plt.subplot(1,3,1)
    plt.imshow(Alo,cmap="gray")
    plt.title("Original",color="red")
    plt.axis("off")
    plt.subplot(1,3,2)
    plt.imshow(Rq,cmap="gray",vmin=0,vmax=255)
    plt.title("EVD, k="+str(k),color="green")
    plt.axis("off")
    plt.subplot(1,3,3)
    plt.imshow(error_image,cmap="gray")
    plt.title("EVD Error\nFrobenius Error = "+str(round(error,3)),color='blue')
    plt.axis("off")
    plt.tight_layout()
    plt.show()
#evd ,images are printed/
#graph--Evd-/
font={'color':'green','size':16}
font2={'color':'blue','size':21}
plt.figure(figsize=(11,7))
plt.plot(k_values,errors,marker="h",markersize=3,color="orange",label="EVD")
plt.xlabel("Number of Components (k)",font)
plt.ylabel("Frobenius Reconstruction Error",font)
plt.title("EVD Reconstruction Error vs K",font2)
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()
print("Executed successfully")
