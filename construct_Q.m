function [Q]=construct_Q(Dlabel)

Q1=Dlabel*Dlabel';
A=Dlabel'*Q1^(-0.5);
Q=A*A';
 

