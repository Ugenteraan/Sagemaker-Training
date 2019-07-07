import sagemaker as sage


role = sage.get_execution_role()


sess = sage.Session()
account = sess.boto_session.client('sts').get_caller_identity()['Account']
region = sess.boto_session.region_name
image = '{}.dkr.ecr.{}.amazonaws.com/quiltdata/batch-test:latest'.format(account, region)


clf = sage.estimator.Estimator(image,
                               role, 1, 'ml.c4.2xlarge',
                               output_path="s3://sg-batch-test",
                               sagemaker_session=sess)

clf.fit()