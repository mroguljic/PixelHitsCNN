
I
input_1Placeholder*
dtype0*$
shape:���������
�
+model/conv2d/Conv2D/ReadVariableOp/resourceConst*
dtype0*�
value�B�"� ?�;�>�
�=���<y����8���Sw=�b��T�#�@_�<�x�<q>>�?�~�����=P����&>e�9>2pｕ���Q=7i>S�=H�� |���V?>.��=� 7>dUZ=�LJ>���=0��8�h�.r���郺u�<�2���;>��t=��J�k.����H��.������Ͳ=�d�=x����	>�%+;��A>�[G=00����*��޽�}@� ^�:�[=M�>�>h�+���F��8μ�b>�+��xv�P?�<<��f� z��B�ݽIo&>; ��Ͽ>��<>Z�=�\��c���Q=��7r	>�Ň���W;�%=*�E�~�½�A���=1����<"1'��7?>�̋�������=+�3>?�.>p�����=�8�=M������p*>��<�V��v=�й�| ��`�3=�Hn����<l)>�!D>t���SM%>y�(><?R�ޣ�=sAA���=jm�=F���x
>��x��}�=HK4=��v��b��f�(O����<>�e˼�)>8K���,?>0y߽w��&���Q�=������-�F��=)>�?>
d
"model/conv2d/Conv2D/ReadVariableOpIdentity+model/conv2d/Conv2D/ReadVariableOp/resource*
T0
�
model/conv2d/Conv2DConv2Dinput_1"model/conv2d/Conv2D/ReadVariableOp*
explicit_paddings
 *
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*
strides
*
T0*
	dilations

�
,model/conv2d/BiasAdd/ReadVariableOp/resourceConst*
dtype0*U
valueLBJ"@                                                                
f
#model/conv2d/BiasAdd/ReadVariableOpIdentity,model/conv2d/BiasAdd/ReadVariableOp/resource*
T0
y
model/conv2d/BiasAddBiasAddmodel/conv2d/Conv2D#model/conv2d/BiasAdd/ReadVariableOp*
T0*
data_formatNHWC
<
model/activation/ReluRelumodel/conv2d/BiasAdd*
T0
H
model/flatten/ConstConst*
dtype0*
valueB"����  
c
model/flatten/ReshapeReshapemodel/activation/Relumodel/flatten/Const*
T0*
Tshape0
��
*model/dense/MatMul/ReadVariableOp/resourceConst*
dtype0*��
valueՈBш	�""�����:��<���PT�;r����2伐���3~��C<���#�< �ʼUμ�r�<��<��t<��{��2A�<(��}Լ���<h��;���<��H�޼�<�7-<���<F�<6<0eS������<\=PM��������<�d
=��d;�H�P|�;`?���z=U�ϼ���$��.=��h<@*�b��<��=���:	��[���'�ب�;R""��m��;�XH�;��F���`7�;�d�:�|<F,�<R��<��ü_�=C>=��1<0Bx<�r=���<(��;Am=5	=��H=@*X:�!<����<m�=�VG<�w�b��<,m�<��@�:�!�;���;���;@�1;���@*n�R�<X턼��&�p<i< %�9�s��/=��<.K�<�~E<I�=`�;�v�<����Hlϼl��<�F
�� )<�/�x��<'���r��<hZ����=(N;<��M:�M�<�lI<Pe;���M� =�È<�@�<ư�<2��<��<����	���Xۜ<���<���;�;��_�<��ͼ����0��f������r��P�<�Z���H9<$�<P��<���@@,:oB�a6�N��<�<<��\=�H�<p��`l���c�<R-�<�T�����8z�T%0<�l�<k/�#	���<��;�!=<�y��8Kջ�7b��A��PI;�����y�<�e��A�<�<�)�<|���=�@�]<�[�<���<�I��	��4(&<@󼸯h<�.��0R��V�e�x8��<�<��;�4��N��<l<8r��@�:���I��!��j��N1�<��B��v�<4z	��� =��S��kn�eG���=<�v����<���'����=���p�9;Eb���v�<��x��-=�p�<.:;��+=�X�>�G� ����96<P&|�� ����
�����pE`;�aY�s�̼ڑ@���<Eq
=B/ �B��<��;���3��9�=X��;�S����=�)<�����~� �U<�T��� �@��;j��<�>���;��ޕ�<��ܼ�Sͼ0�T�&��<�9ͼ�Ӽ�W���ˋ<��ļW�<@�<y�I�<���d�<�<�;��M�w��,Zs��Cc�.{<��;1�<�=؁<�۰;h@<T>�x ��^���_ɼ^e�<��T��|O<�w�<���<P �<ft�<PC|�g�=����{�=T�|<�k��莼���<\�6�&�<�؉:Q�=@�:�����糼|�� <V<0��<c6��@�H�m~��w=HJ�;�Y(; �L����������F< hW<�O��xnڼ2��<j�t�}�=���ʚ�<�}Ȼ���<�M�<B��N��<���<@gs�PU�<@Z;3����=�J��켊x�<5���=�=z��<}���TW��\ϻ�Ȑ���y�Y�<�� ;�����	=�ɩ<�q4�p[�;alͼN��<���9�=��݊������<�P=�̻<~<�ż闼<�����	=�����<R��<��];p���k���<h%��=
�< `�� Vc: x�:����y	=ʖ�<��N<���< ٭����<�*��G�=&*�<�cp���c����
��<�J=��<0;�B�<l�<b�<j�ɼj����'=�@�:�؞;���� �<LC'��������<�5 <(���d<���< ؏���ú�~{������G;�tW<^��<���t�l�)<6��</ ڼ�c�; |�8"�<T�����=T7�Y|���<PZ����f<�S������^����<�P��F�p?�;l w<(Yy�<>� t���p�<s����u⼢�<E=�4fN<��t:�`�^>c��fr<�u�<Lk1<�ť���a<��
=@{��_�=l�<�PK:�̼h�;Hǧ; �D;�.= �A;�S=�<���<��Ļ�p� ~ظ��ȼ�Aɼ`)���P�;���3���7��mg<PȨ<U�=��=H<��F�����<����J< �(�����K<��!<.Sü���*-h���D�����ꁪ<`{���۸;�l����ۺ�OS��#���w׼�<P���VU���=21�<Z鱼��#�����\ȼ���<�U¼��q<�r�;$Ⲽ$��ǋ<��Y��}��Ἒ��㍼�M���#�y���^=���j��<���<���<P��;��=��6��z��<�fW�"�v�(η<(系�=��=�R�;G=߆�S=�ɇ<�x���� =܁`<���B<@Tڻ�O���Ƈ< ��:�=z=�<z��<D�S<�	��B��/�����n��<1@=ȯ<·A�e�<#|��
Fy��4��U�<K�������C�<�=�=�<;��"� 0�9X8�6��T�n<�&�O����)=���f�<@)&<�SV��뻢����`�;6��<8�f��<]�=d�>��6� q���|<�(�<L�弐+�;���@�:���:`d*<Iu��rG�<T�μH/�; 
��r��<��u;�J=�Y�ڋ��p����_��?$��#�����<(�<^�<D��Cb=h�ּ�F���`f���<<~M�<�$X<~��<`!�;{= ꧻ� =w7=���p�<�#U<U/�^F'���=�ɍ�α�<h�c<�u�<H�<�Ѝ< �&���H<��< XM���i<�(�8�^���=�Ҽ�݌<��+;�=RO�<��(�<l����;�h�<���;��;�Ō�������J<���t�漢���9[=r��<�<��=��ļI�=�1��,Ҽ>�	�2��<q��l��<��<j_��@��;N'�ܟ��
1�`X�<�M�<@����c���h<�b�����<̎F<plW�����<n��<��`ʈ;��� ͕<�
��de�<�y��bi�<�����}�f������:�=�]�;�!�<�K<��}��^���x��x1�P�(��[o<@N&;�N�`}��^Vμ�]���8���;����,a#<�*����<P��p��;@�f:��<�'��H_��ͼ��ļDki<�l�<�:�Ǹ�X�[< ���0[A<�x<����������	=�|�<�_�<�@�<Dh�@*�;����ļ�M<����Q�}-���lռ87<��><����=_���X�;��<k��ܱ<����m���=萪<R��<����C��P�����<�:�h͝<�=ռ���<(Д;."����ͼ�K�<V_�)8ua��W�����<R;�<Zq�<`��:ʟ�<*Z�<ă< A���D��Z�<���< l�w��d[ʻI���U=����E ���ڼ�i=��=8����W�X��;�;\<������f���޼ʼ<�£</� =4CD<�S��ο�<fH���Ȗ��Bꉼ�����Z�|���x<�0=>ɱ���;P�񼘭û(�5��77:l�}<9ݎ�~�Ӽ�N� �`pG<�����Z�V��<��E<HҾ<_T=HQ�;�����P�M��`=l�<V�Ȗ(<�R���W��=<�(=ޮ�< �:HW�;u�<�����=��}�4�?����e���վļ0��hԃ<�`�<�\��q��Le<�`�<@�<$V0<l��<�od<�}���RK<��ü���<��;Ѿ�^��<t��)�;���;`�v<BΩ<`�r<�7��}A�V��<����X���\k�`pq�&�<h(F���<\�ؼXr���j/<��ݼ�o�<^��Z��<������<��o���漻����-�<ꢃ<d=�<�R�<���<�Ao<�x<��G���ɼd= ����笼�?	<�s��c�<�ݯ<��<�|�T�
�<i��fn�<���< h 9�C�<�
= t�:����t��"p��O�<Ht����0��<�<댃�y��h>�b'���;�=�͓��&�<��Ǽgټ��Ƽ��=�X�<43�<�v�����0�r�s�=���H�;X#��ց��;" =�� ��C�Y���$�<.k�p����g�.Ҽ�;@�:�� �P�p���뼦�e�t��{�=�p���G$��6t;>\N��	J���ڼ\�;��x�;f�o����<�g(�~阼����� l<(5ۼ8�<����
�Ҽ�d	�ܯ��_޼p�	��+������9��8z���=@�<(��;0#`;^8�<��F<�$�;����<���<����)� '�<8��;���<B��<�<C%<(%���++�`��Fv�<�Ö; ���� =�F���M	=�h�9狼�J�<"��<S =(J�-���m-<yg��j��<������ɼ���<x��p9��d�<^��<`3��)l��Ym	=@m: շ� �G<�����]<	Ѫ�lQ���i<ռ��6�rk[�8�<D�=�8�м�#
��jy�fwѼH��;�����<P ��{���������<�:�;�U<#�=��H;�~�;��C�D&��� �0<���;���;@�:��<⚱��ۄ��2��Z=�ɸ:҆�<�!<����nH�bkռ]V��~K_�qջ(	�;���<�<N�<\���ԧܼX�i<4�-�+I�����r��h���җ<� �0��;z��<P�ʼ8�
<hR�<��W<t�<b72��RP;�W<0�ٻ��?��
��4�����*� ��<Dἠ�.;�<:3�Z�}|���� :n�����<���< �� Yi����<*"�<�V��V�����Q<���<��ɼ���<�W��x黀���:��Ē����.��<��;D����H:�Pu<+|=^��< ��;�=���,g�D�_<X �<p^ʻ�1�p):;�x��_�¼�"<:��<v�<	�=@M�;(ﴻ@|&�<�����ɼ�����<����X1<l���S=qR��hv������ֺ N:0�弖p8����<`٤;��Ƽ��ݼ�H�<��D�8��;�ؼG�=�W���˭���<hY��O�;���:|r\<�~�<r0���$;2��8�i�ca����<�=���<���	�;M4��i���ݿ���μ�X��܃�&��<�|����<��.�������=��.�@�<O����</=)h
=�_4� ��d�<�e����vT�����<0=�8<�;�F�<��� �<�W�;�.��&߼3\=��\;^}ټ\X�< r�4:?<�*=�A��h�<��q)��=�.� �9�K<�9'��H�<JK��(�-<`����Ӽ[n=b5�<���<�l���D:4Ⱥ<>	�<��<���;�Xɼ�� �z@�s=^��<����?� �e���@s<�n����<L�K<��<^��`F*��ڼ
��< P�9у=RN��ͮ����;gռ��<(z<0����=�Z������<'=��̼:z��N����wF���<|Bx<v{m�f�k�P�Y�P�{<�cB����b�|�n<�31��|5<b_�<����̼��<r��D�|�p��;��H<Fq~���= \�;�N�<��|�=P��hŻ�#黦���u�=`�0<���_rɼ��<@������F�<����H�<v�<�?¼���<�!<�i�<��<_�=�i޼�2t<�;l�����{Ļo�=H ���=��Ӽ$8O���Z�<����f��<���%�n=�޼��ݼ��W��f�: ����[0���=�ds;ج_<<
;�a����:���<ʧ�<�<�)�<`��%����{���`�;@}�<�T^<v.�<T!<\��<�޲;��Q<�z�;  �`h�<P�J�����<�����G9�0"n<H��;~Ӯ<�<�<���x��\?,��Dm<X򺼕U=���<h�`�;�s.:��; Gҹr�t� �
<x1��m���E�<N`�<� ���75;Pj;a�=(H�;)�d�C<ڤY�V4�<��<ܾ`<�^0<l�ʼ?=@��� 
�;�s��z	�����@����^}<���<'#	��8$:|�6<��< !�<�<&��<~��<�4��.�ƼR��<�&����[��@�9x�T<��<�U�;T�N<���&��<��=��{�!�=Hù<8�;i�����+<�ȼ�ҫ<�:4��*Ǽ
��<0��0y�;@T�<���<(	�;�+����żX�<�	 �(^�;�(=��<'��ЪỶe��cx� �Y��e=��=���<�׹<�W�S��:��<�<�d�<�-���}ͼRތ��e<��X;�ձ<��;Zo��0���Z<j�J�@9(�� <_R;��L;@�g� R��|<Bq�<�7z<��e� ($�w�<������h���;�y<��<-�=|�<dd�K����ż���<���<�t�<��=g���b+�(����q=��0;[~=����/< �ɼ ��:���<1�=�y�:��?�/,��[�=�<7<��$<8ٚ���:"PG���;h�+� P<Y5=���<e=�=�j��<<��<�:�;���^��n��<Z��<:��<�T=��=p��B���r=[湼�(�<���<��G�������<�m��9�<<ڼ�^=H��2*�<K,= %}: <ºX�d<��;PN5<p����������7:�n��&�ǼR��<��<!ﾼ��:�8[�<[�����=�p�<ڿ�<���.�<��.����|�?�|<��%<�|�<h1a<�}="��gV=����lռ�:�<C@�_���j<�%��J=��=�b�DDּt��<L������Na< �79�������<4�#<��꼊��<ު��#Y��� <��
� 즺�@=/d=xG<�L�<(	<��f��c �,�<`'[<h�׻u	=u���i<��J<Y���4��<�縼\,:<���,ۂ<5���j��< 8=�|�:��B�.����s�;������\��r��Y%<��<�<��0��<�vA<����Y�;��<X_<��= ߍ�$E���d�9<�&DP��0h�`��;UN�-=�y�Ǝ�<�O���<B~�<KM���<��?<��]<��H;��
=��< g:\�O�r���@!Z;���#�<�N�<�j<Lzs���;4�<``';���<�c���<�>&<A=7�
���=~S�<L�<=4d%�,��<��F�<�iҺV��<N�ݼ�0�<F��<��b<�,�;*$�<`� �$;��;���:j�ռ��<����<n< ��:̬z�Z(�<=i�������O�ԃǼ�^�����<�m����¼�����v����=h�]<@�z�������߼*ϼ�{<P>��(�b��֏�<�j=~�<�TK<�J��ڂb�fC�<��;|��%='���V�<��:@≻'�	=�-��%�<s� =`�<���<��<mI	=���<�ͼ'=�J�<0F];�n���%I<�\��8=�Ӊ��O=��`��u�<Ŝ�����<y疼�cռ����<T��<*��<M����͗�r��<|���;��Ĥ!<�E=81t<��;�:=r#���
���E�2L�<�������6��<\��h���h��詻�<�Ļd�#�������<�;	=h��;6��N~�j*�<$���V0<���fG�bAȼ��� ��;(�r<��>�|��P�<�B�<t��<V���B���ʴ�< ]켠�庤z�<�b�<X��;X�<���ڼ@+;�p��R�m�����@�׼0��;8����j� �'<����X\�; ar�H樻��=���`�;���<��=W��d7�<$IԻ��<�*=��<�oʼ� �ź�*�����ݼ&���c��9Z<י	=��<�n�:�=����=��;����=B��<Z��<���Ԟ'��+�:6G�� �<F!����Լ�I� ����~�J��<�G���� �<�鳼�;��<�~�+�=���<n�ļ���<��h���X�3<��<4�<e��0=�7<t+�<�G�<��	�s�=>��<b�<B<�<��<�+	<��=�|=��]< ���K�;�ֺ<.��<zob�h?��7����<E<���<��?�`��pg�<h`����:X�¼+�=�X:"��<j��	\��!� f�����]Ɵ��~�n�x�F��4[<p�4��i6��:�&������� ����� q�; ��9 �9�N��< k*��=0l.;�<h=2�<hɲ���
<|�(<�z
=��Z_�<�ü0YO;��|+ػ𗁻�9��>Ҽ���j��<�߻RM�< ��H�c<\�D<Н�;���<���<0�<������E���:� ����T��w�����t<��Ȼ~�����W�Ƽ�݇�F��<�֠��x��<�V�<6ܨ<�9�<O�żۧ�><�;�<�jt��H����\< _9�ػw<\�P��6=vi�<�(R;�<��<8�e<�O���rӻ��=�E����<c�@?<U�=���;�< ��; р����<pU<��������� �d<ˍ���廔}�<Z��<��<�Ui���)<�\j;̳���� ��:X	ּ���:�&�<#�<|�<��=(��<�E��S�<(N<�K�<�P<�����s�F��Y=���F�Ǽp	<P��<R&� }�.���������<" W�:��<󝃼�=��=����P��;�=�a�<d�ϼ����1]<q�<�Ҽ�?;
��C���jM��!=��@~Ժ8-<�	м�W=�BЋ<C=��=<��= �9��;?5 =�V�<pj����;����赼  q; ~�<h=&<�o��؅m< *Z�@���X�K<�<�;��<�d�;�dѼp0�;_b�O@��/m� h�;�/����@�Һ�� �b�	��=N�����$<�O
���<̀	��B����<~�P��H��M�</0������0�?<�"=�=pY���v�r�X]��]=�oN�\)��:<��i�P]ټ<�ʼ�)���c=h!\<+��P��Е��&Kz�������<�~�<Z��<h��"��<�ʲ� 2<d�#<J��< �E�}c���i����<�@=������� ��WU�沐<2�<Xf�;��P<��ȼp��x;t���4�摅<���@�����n��<S���{�2��<��<L�V<-4�����;�<���<8�4<�ʀ<܀�p+g;@"ܺ���j}�<������u;����B D���<P}�<����< ��Bi�<+����׼�t���<!�μxy缙�=�=Ě\����;p d<`�ܦ<l)�hA��/м��dB<�����������j�<ؼ<P�<)���=��1�ἄk<�<B�8�r%��Ӟ<q0��B�W��d���C<�P�;fPؼL�J<���<J)u��j¼,�x��Ỗ�<��-<b�<8-w<j���������;Nj-�>��<�9�<`1�; =&��<��<9�=
;�<���<m���XՒ<�V�<2��<r��<Ȧ��R	��#�<<�/<D��<�9J��)�< R"���ƻ����6ļ7gռ��<P�;Lfм��ּ'a
��=F1���S7<cZ=�N輾�<�<&����K;9\¼ к� �<ص� e�?��� ��8���:
躼�Y����<��=Đ��r<�;��.��<������<}Ӽ�����U��������~��`��:Ұ�<�%<�^=�漥���<%�< �� �ӹ�����a�x�V<T=<�=�<.$�<@:+<�;��ռ �_:�y=z��<,K<�,����<�w��'����j���S�<~5��E�<���;RS�hN�<��<��+�q��^t��`A�:�q��(�<&��<�#<�2�·�<v^�<'G=.<�<R�<�����:�����<�=hx�;r�<�O�<6��<�:;�%<���9�=���<�� �};._�<���<��<@qͼ�ݪ�h�X<\�<x��;��$��<pl�;X�I<t)X<&r缬�8<Z�<����J���F�<���q�<t�伺��<�>�<�%a<��<���<���<�!Ǽ���;�����sp= �g<�T�v��J�B� u⼘�üN@�;T�<<VڼGP=d�ʼ������;hq�<ɟ=��;�P�<�K�<�I�!R	=m����2<$�����<��x;�N�����=�ϼ~�ɼN�<f��<4b�<[X=�r ��k=Tr<���]����F^�<�2z�m����U�a�<���� 
����0�<"d:���<	��6<�9Ӽxʯ�(�P�:<u���;��� �s��4�S��p��;��Ѽ:ؼn��|��<�<H��(�i<(�˼�����:��=���; �V;P�`<�?<�0�഑���;��輸F5<��<l;0��j��aί�`C@�T=�f��<\�E<��u�5i��X�<�=��0� %��` E;�&�^�<��<�`i<�E<�<�V�<�wH���\<>��<�k����<���r�� �r;�D��K;��<��׻_�=�(Ӽ滓<����t�u<9�=CS=�Q����T\<Q<3�=%w;@��;��<�U�<����(�1<H����^�~��<>-Ѽ�Ѽ:�<��1��lS<� ټ��<\E��I����<M3=@c�;P�i�8���0�׼��
<����û�ح<Ćn<jH�<�	�P�|;�����ռ�Wj���л��Z:0����ߝ< k%�Ƴ9��j;�py�D�<X�y��<@��ۦ=�����^�����!�B���xz�;'�ټ��p� ]�9ץ��$N�<�����kռ8��;�/��X�;����F��<�j�<58�@!�����&�<_q�������<K��\B�<-��f^�ipܼXɆ;��&� ׎<��ݼ`�9;��d���<��<����뛉��?=D�<ʍ���D.; ���L�i�=��=�H�;���;�k|�Ԗ�< �<�9��+�O-�R�<�=�����\������S���<Sh���>�<+\=������+��ZǼ���<���"K�<J�üM��KZ
����<ִ�<��%<*��<�u<�o�<�I=�@���=�<X�<<��);���<�� <|�̼�5�<ps<�y���.=��A:HĚ<�e<��6�+ϻ8x�<H|�;�<�Ǽ�{�<#(=���r�<�2�tJI<�.<�����<��ȼ���c������<z怼�@��`
����dw}��3�<} �N�<&��<���������
��
= U���K�<��;�)=����u}����=�6����=��:J��<n��<��<u���R �b��< �b<�Oۼ�o�<F��<uk�r�r��f��O<df#<��;�����<pI�����L4꼀а���n<�7;&�μ����4� <Bu<���<B�
�~����Z
<@�<�3�<bU��Pn<X����b<���`��@�W��'d��]�< k���B�R�����<����c�;V�"�&�<`��;~(-�p2�<vp�<g�=������<�@o<p�˼��<�Z=pN��ά�<�������{u�LU< �X����9]��`�ۼ�!=����+�=��h_H<�0<��ļ������,�i<���P��3=��[�ZZ�<��<���<\� <;J
= Bd�M�=�f=xNC<�n�<������<\w���J<���<�d\��%��\�<���; ���ju�@Q�:L߼�{��z�� �Q8���<�J-<pg��r0��u�;_�<'�Ȉ<�S����@ZC����|�+<l<��<z�^���;r>u�~R�<��� #�D�ջ�6�<�J�<�~���<�K��j���J���{�<�넼T�)<����$���NӼ�	���^<.��<��j<��<��T;�X�(�_�\�`<.߼�D�<��?<�*-�v�<�__��a�<��ɼ�����< �����<����p��;[�
=�L���60:*�����;$q;<�x�;J��< ̣��[�<�|ϻ�Wɼ\W[<�
�<8ܥ��ù<fE�<� ټ菄��V����<�T�!�=nр<�1�l���� �<�i�T*3<���<�*���yH�2�����<(<�;���<T�<�.��f��<B��< o�<*>�<�l<(�_<��py� F��8�<a=���=�d˼�Mżf��<�c�T�Ȼtw׻��=bL�<`��:��<��<搙<W$��i�=��L�~��<���.��<�
q�l�»�ټ�Df��c=� f��ջ�Dv<sF����E���B��Y9<�9�<��7���"��<�������<��<S=��Ć[<>�����<�.����*���Ȼ�
=�Z4<�Y��NQ�W���?�<��<څ�*w��ʿ�� �<HX��N	=�$�<����
�K�Bd�*>�<N��<�W��Fg�:7�<K�=l�ݼ �t;�ܼ�D`< �����p<��ú�<�vx�<0><>Go�<2-<�瓻��>:0һ+U=f�����Ǽ�ͣ<��=V��<�����Ź<C� =vڅ��;,gv<�<���<��= �<�q�<*��<�����=�="�X�z<8z<<\z�<�
a��閻�6<J<�<���<�=���д<����S�Y��~�$�tXR<*͕�pR�;.߉<�k�<�x<d+��΋�<��K<�E�<�Rd<x�%<X�(<���:�m�<|{K��Is<X]���?<qz���c�;�H����\1�Mެ����&���j'��Ҽ&��<#���Ԝ=<� �;�0�P	��Q��"�l��`<�υ�ز�ܠ�<�h�Dk<��/��t�<�m=�_�!�=�� �h�b<����=��� �<��{��d�<<������<Pҳ;�ll����< � �I;L*\<0>;��e<��4�<��Z���#9����<�Б�0���n��<�K�<.�ټ��@R[< S�<Z'<a�꼮� ��J=9����������<贵<��<p�}v=���<*wW�Z�<��� ���wd< /�:" �<��=�Hz��怼@r�:6d༄�C�u�; tR8�R<�ؕ�@y���<j��<d=���S;����=��8�>��<6h�<n(�<*�<x��;��K<m��������=PN�;z��<�2=�T�<�Ϻ2#�����<t��D�`<M<߼��'<�-=���s;t%�<���aJ��-:"���^�<�=~洼�MŻC���Z��'�<�˛�����'<`h�<���<#���k�����j��<�n^<x��;�J�^�<Q�	=�r��-���9ɻ(�W���<<�<:��� �A<�;�< ȟ;%���j� [����X�������<���zZ�<0&��N���Ə����';h�<$V�<"H�<�s=kz<��޼�����v4�;����L�|�d�`������<�a����:8Е<���:ה��������<��p<�ݓ;��:9����`�<LM��x`<{���Z��L$ټ�<d݄<>V߼��=�kμpQA<b���19;�"g����6��y: �K�D�&<`�;���A
��N~�<�G���=/�;@�<P��q =��_:��<��=*	�<�K޼cl��:�<����%�r�<��k:�.�����<�\ϻv[�����������1�6
�<~�<(d�;Hsg���= �:������o��gK =+�<�p<P�;v��8Ƽ����f|�<���:��J������`�<�q�H'9<��P<�@Ļx2�<�d=8�;�a��z<��ʼ�+���.%;C=~cɼ��� �9.h�<��k<����]<
�d��3�ٿ=�0���.�<�C ��:����<��** ��*�ڒ�<���t.��' <�w3<*�ż�m�TA��*G�����0Ī;�^�x�W<�]	�L�<�>�����ޱ�<P�O<���������<�R=,��lg��=�~�tڞ<6��<�H�<�����S�;�߻��I< �7mֻ�JV��x<|��K0<�g�<��~�ԼKr=@i���/<Q�ܼ�w�<���=�<��=�hҼ,��|��9�
�8kڻ,�ż ӽ�(��; cK�!ő���C<��;@o�� �Z������<0�n���=FS��DlB<m�缠��;;t����/=��<������� P]7��v�x��?��v�<�B =u =��<�n)< :I<r5�<Z]&�2�/��[̼C;	=�<ܺ
��7���<kk���;�N<;�	=Tb3<h*� xy�z�<���<���""�vܰ<ʱ�<�x�b��HO�;�	�<���M�� �	�Pd:�l6��l�<f���7�<0������<�|��n&<,h���l�<�{�������xt�;��<ւm��A.���<���FW��=w<¹��hY<�P��fv�<7e=����b���O<�L=��=A2��9=��(<�B�<���< <�\� <�K�ئ�n�� N<��e<X҆���<�ܢ�p���Ƽ�����7����<N���H� ���@�F���0<ta,�Ht��kU����<��� �|;�%<घ�� �4ͼ������3�Ǽ0\<~D�<����xF����<|$���<!�
=@��;_��M�<2������;`{J< ��:�*�1�ʼ��<��`�������<.�-�,>�<���+<��<l@м 7(���=���<���<q����"�"a��#%;A6=`�;�uǼم��{�\<�t��<n<�J���x�\�#� �Ѽ�"���S뼠���d�A�=nI� �;2��<�c��}��� �i<�]��64�<�X��r����˼��k<ǅ=�Ӽ"�x4�;j;�<д;��=�����L<�f� ��8�� ��O��$q<��8G�<��8����.��U���o��D��<F��<^��<���<@�|����<G��N�;�g�;i��L'\<��<0s�;H�;�M���$ =DF�<�w� �2�м�B[�je��6
�l?漐0��<��<H��;�f�<�{޼&�x��=.�W��(ڼ�pж<~ͬ<�sμZ+��3=pq ���*���<-����<�r�<h��;�ݩ��h<�*�<a譼��G2�̼0:��y�<�ٙ�����bP��|���-�@�x;�i���p)<��^�m=�<��`h �V{��jp�<��>����2P�<�7V<f+���<�ڻ�=h���8����<Pw�;hR����������b:T�ɻ����O�<���<���WH��I���������v��<<��I����)8���7<�S�`q�;t'�<0�;�ݨ�4c`<nl�����<���&)���f�w�� N���:�	=���<ԭԼ�����i;`�� O�;�u�<(�@�L:h�;F��r��\��T΄<$��<��p��-�;ڸ�<���<��<@|����hO<�	=D�B�׼@�=<����N�\ռ�κꗼ�.=��ۺV��<����)�*\��9'�p�v���L:��<���;!K��`W����ڼ�s�<v#�<�UD;��<d�<^� ����������������[�<1/=Á��\����H<��/����;��8<2��<��<$"H�A\=�̱<�λ��	�@o%� ��<ة߼��<h���������`�����m�=0��|��<F#�< ���Լ�<[��� =�9 <�d=.c<�ļ��=�\����<Δ��,
��ߞ�&.�<���<p��<Zv�<�W���l�<ŕ�PX���ټ~���B�<�N� �:P&f����<p�0<2"<2�Ӽ"�<2�ʼ��5���,�r1���;�7�<!.��w]�y�� 
G�H|+�`֞�>��<w5=��Ƽ�~0�do�<h������a	<F�P������ﶼ��U;��i��`���6�<���<lx<�B�<�"�<u���[	=���<PY�۟;p�ҼNS��Z�[��%<.��<G����
��ВQ��d��#"���<\u;���0����!����T����B<6d��� �<���>��<�t=Vj�<�������F��e%��!6�pۻ�=t93���#�N�e��p����);����c�I�=4����<3�=(�뼤�6<'X=΀�<�ɬ��Ҽt��<f�3�L�<��	=��;��<�|˲������7%�Z�� �<��<�f�:��<��<�f��U���|F;u���S����;j>�<��ӻ�h�<p�����<�<W�<d�h<i�����<�8��my����<����̫��ݼ�>������g�<t�%<p(�*��<.X����<�D��P޼7�=ʾ�<Pl)�X��<����"�<���< П�l�O�=l�޻ē2<���U޹�a��C�=�y������i<F<�!?<��"<�=n�q
����=�G��3x��D���'=z�׼��<�y�<h�;ڪ�<��9<p�t<`�ۼl�<��:��;xz��f�<�Ѽ-޼���$\��=�;�����IF<��<
`���ʼP�q�P1��üx��Tyټ��ǻ��6��<<�<�g�<�3����<�y���=�Bh�䆇< ����=5X=F@�_H=|\E<������=��<�CƼ��o�xb��	���~<�}��މ�<_�=�ɽ<�=-<��E��E�<b��<�q���=�w�<��k�jr�<(��̑�3�=-rټtU<��Լ��r��<�	�����8�<&ʹ<<��hp����<�� =Q~���<���1��X�< C9(�;�.Ӽ����P�n����Lp����<Hc��@Z�;�,�<��I<�o������T< Y�;�ټ��F��j=�d{���]<콦<���a��
~��<>��<
�<�K�<T�<xD����:=��0rW;�����w;�}�<"�!���;�<	��'�<��ü|L<���<���< ��;�«����<+k��G��
b
!model/dense/MatMul/ReadVariableOpIdentity*model/dense/MatMul/ReadVariableOp/resource*
T0
�
model/dense/MatMulMatMulmodel/flatten/Reshape!model/dense/MatMul/ReadVariableOp*
transpose_b( *
transpose_a( *
T0
\
+model/dense/BiasAdd/ReadVariableOp/resourceConst*
dtype0*
valueB*    
d
"model/dense/BiasAdd/ReadVariableOpIdentity+model/dense/BiasAdd/ReadVariableOp/resource*
T0
v
model/dense/BiasAddBiasAddmodel/dense/MatMul"model/dense/BiasAdd/ReadVariableOp*
T0*
data_formatNHWC
�
IdentityIdentitymodel/dense/BiasAdd$^model/conv2d/BiasAdd/ReadVariableOp#^model/conv2d/Conv2D/ReadVariableOp#^model/dense/BiasAdd/ReadVariableOp"^model/dense/MatMul/ReadVariableOp*
T0"�