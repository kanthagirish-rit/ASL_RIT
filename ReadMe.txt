D:.
│   gloss_frame_connection.ipynb
│   gloss_target_conversion.ipynb
│   key_points_42.npy
│   key_points_42.pkl
│   key_points_5.npy
│   key_points_5.pkl
│   PCA_5feat_sentence_frame_mapped_l10.npy
│   pretrained_155_f5_v1
│   pretrained_274_v1
│   ReadMe.txt
│   sentence_frame_mapped_w1_42.npy
│   sentence_frame_mapped_w2_42.npy
│   sentence_frame_mapped_w3_42.npy
│   Test_Encoder_Decoder_gloss_level_5feat.ipynb
│   Test_Encoder_Decoder_sentence_level_42feat.ipynb
│   toJSON_PCA_5feat.ipynb
│   toJSON_PCA_m.ipynb
|   
|   (Extract all the OpenPose Video JSON Files here)
│
└───pretraining
        preprocess_pretraining_data.ipynb
        Pretraining_toJSON_PCA_m.ipynb
        pretrain_key_points_42.npy
        pretrain_key_points_42.pkl
        pretrain_key_points_5.npy
        pretrain_key_points_5.pkl
        Pretrain_toJSON_PCA_5feat.ipynb
        Test_Encoder_sentence_level_5feat_all_train_pretrain.ipynb

        (Extract all the pretraining OpenPose Video JSON Files here)


To run the seq2seq network directly. Follow the below instructions:
-  To train on 42 features and on True english translation, run 
   Test_Encoder_Decoder_sentence_level_42feat.ipynb.
-  To Train on 5 PCA reduced features on Gloss, run 
   Test_Encoder_Decoder_gloss_level_5feat.ipynb

To Pretrain the encoder:
-  Run Test_Encoder_sentence_level_5feat_all_train_pretrain.ipynb

About input files that you can use to train the seq2seq network:
- sentence_frame_mapped_w2_42.npy, sentence_frame_mapped_w1_42.npy, 						  sentence_frame_mapped_w3_42.npy
  All these files contain selected 42 features of frames for corresponding sentences. These can be directly used for training purposes.

	  Note: I have not generated frame to gloss mapping for 42 features. If you want to perform it, bu running gloss_target_conversion.ipynb using 'key_points_42.npy' file.
	  or 
	  you can run contact me. I will explain on phone or video call.

- PCA_5feat_sentence_frame_mapped_l10.npy
  Contains PCA reduced 5 feature data mapped to corresponding gloss information. 
  i.e It contains a list of sentences (length of 10 gloss words), mapped to corresponding PCA reduced 5 feature data.
  These can be directly used for training purposes.
