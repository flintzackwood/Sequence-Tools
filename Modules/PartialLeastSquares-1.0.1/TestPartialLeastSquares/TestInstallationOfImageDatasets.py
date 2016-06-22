import PartialLeastSquares as PLS
import unittest
import glob

class TestInstallationOfImageDatasets(unittest.TestCase):

    def setUp(self):
        print("Testing that the image datasets are properly installed in the Examples directory")

    def test_installation_of_image_datasets(self):
        items = glob.glob("../Examples/head_pose_images/*")   
        self.assertIn("../Examples/head_pose_images/training", items)
        self.assertIn("../Examples/head_pose_images/testing", items)
        print("\nOK\n\n")

def getTestSuites(type):
    return unittest.TestSuite([
            unittest.makeSuite(TestInstallationOfImageDatasets, type)
                             ])                    
if __name__ == '__main__':
    unittest.main()

