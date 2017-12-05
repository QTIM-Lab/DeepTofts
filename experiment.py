from model import Model

if __name__ == '__main__':

    # Recurrent
    # test_model = Model(max_seq_len=65,
    # patch_x=5,
    # patch_y=5,
    # num_classes=2,
    # cnn_features_out = 32,
    # num_hidden = 32,
    # num_layers = 4,
    # model_type = 'lstm',
    # optimizer_type="regression",
    # n_samples_train_test=[5000,10000],
    # total_epochs=300000,
    # batch_size=400,
    # display_epoch=10,
    # test_batch_size=10000,
    # load_data=True,
    # old_model=False,
    # train=True,
    # test=False,
    # reconstruct=False,
    # dce_filepath=None,
    # ktrans_filepath=None,
    # ve_filepath=None,
    # output_test_results='results.csv',
    # output_model='model',
    # output_ktrans_filepath='ktrans.nii.gz',
    # output_ve_filepath='ve.nii.gz')

    # cnnRNN
    test_model = Model(max_seq_len=65,
    patch_x=1,
    patch_y=1,
    num_classes=2,
    cnn_filters=30,
    cnn_features_out = 32,
    num_hidden_lstm = 128,
    num_layers = 4,
    model_type = 'lstm',
    optimizer_type="regression",
    n_samples_train_test=[10000,9000],
    total_epochs=1e6,
    batch_size=200,
    display_epoch=10,
    test_batch_size=100000,
    load_data=True,
    old_model=True,
    train=False,
    test=False,
    reconstruct=False,
    phantom=True,
    dce_filepath='../dce_data/QTIM_RIDER_DCE_1023805636_19040901.nii.gz',
    ktrans_filepath='../dce_data/QTIM_RIDER_DCE_1023805636_19040901_ktrans.nii.gz',
    ve_filepath='../dce_data/QTIM_RIDER_DCE_1023805636_19040901_ve.nii.gz',
    output_test_results='results_.csv',
    output_model='./model',
    output_ktrans_filepath='ktrans.nii.gz',
    output_ve_filepath='ve.nii.gz')


    test_model.run_model()