import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Progress } from '@/components/ui/progress';
import { Brain, Database, Target, TrendingUp } from 'lucide-react';
import { HeartPredictionService } from '@/lib/heartPredictionService';

const ModelInfo = () => {
  const metrics = HeartPredictionService.getModelMetrics();

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center space-x-2">
          <Brain className="h-5 w-5 text-blue-600" />
          <span>AI Model Information</span>
        </CardTitle>
        <CardDescription>
          Performance metrics from training on HeartPredict dataset
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-6">
        {/* Model Overview */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div className="flex items-center space-x-3 p-3 bg-blue-50 rounded-lg border border-blue-200">
            <Database className="h-6 w-6 text-blue-600" />
            <div>
              <div className="font-semibold text-blue-800">Training Data</div>
              <div className="text-sm text-blue-600">{metrics.trainingSize.toLocaleString()} patients</div>
            </div>
          </div>

          <div className="flex items-center space-x-3 p-3 bg-green-50 rounded-lg border border-green-200">
            <Target className="h-6 w-6 text-green-600" />
            <div>
              <div className="font-semibold text-green-800">Features</div>
              <div className="text-sm text-green-600">{metrics.features} clinical variables</div>
            </div>
          </div>

          <div className="flex items-center space-x-3 p-3 bg-purple-50 rounded-lg border border-purple-200">
            <TrendingUp className="h-6 w-6 text-purple-600" />
            <div>
              <div className="font-semibold text-purple-800">Model Type</div>
              <div className="text-xs text-purple-600">Ensemble ML</div>
            </div>
          </div>
        </div>

        {/* Performance Metrics */}
        <div className="space-y-4">
          <h4 className="font-semibold text-gray-800">Model Performance</h4>
          
          <div className="space-y-3">
            <div className="flex justify-between items-center">
              <span className="text-sm font-medium">Accuracy</span>
              <Badge variant="secondary">{(metrics.accuracy * 100).toFixed(1)}%</Badge>
            </div>
            <Progress value={metrics.accuracy * 100} className="[&>div]:bg-blue-500" />
          </div>

          <div className="space-y-3">
            <div className="flex justify-between items-center">
              <span className="text-sm font-medium">Precision</span>
              <Badge variant="secondary">{(metrics.precision * 100).toFixed(1)}%</Badge>
            </div>
            <Progress value={metrics.precision * 100} className="[&>div]:bg-green-500" />
          </div>

          <div className="space-y-3">
            <div className="flex justify-between items-center">
              <span className="text-sm font-medium">Recall (Sensitivity)</span>
              <Badge variant="secondary">{(metrics.recall * 100).toFixed(1)}%</Badge>
            </div>
            <Progress value={metrics.recall * 100} className="[&>div]:bg-orange-500" />
          </div>

          <div className="space-y-3">
            <div className="flex justify-between items-center">
              <span className="text-sm font-medium">AUC Score</span>
              <Badge variant="secondary">{(metrics.auc * 100).toFixed(1)}%</Badge>
            </div>
            <Progress value={metrics.auc * 100} className="[&>div]:bg-purple-500" />
          </div>
        </div>

        {/* Model Details */}
        <div className="bg-gray-50 p-4 rounded-lg border">
          <h4 className="font-semibold text-gray-800 mb-2">Technical Details</h4>
          <div className="text-sm text-gray-600 space-y-1">
            <p><strong>Algorithm:</strong> {metrics.modelType}</p>
            <p><strong>Training Dataset:</strong> HeartPredict_Training_2000.xlsx</p>
            <p><strong>Validation Method:</strong> 5-fold Cross Validation</p>
            <p><strong>Feature Selection:</strong> SHAP-based importance ranking</p>
          </div>
        </div>

        {/* Disclaimer */}
        <div className="bg-yellow-50 p-4 rounded-lg border border-yellow-200">
          <p className="text-xs text-yellow-800">
            <strong>Model Limitations:</strong> This AI model is trained for screening purposes only. 
            It should not replace professional medical diagnosis. Always consult healthcare providers 
            for definitive assessment and treatment decisions.
          </p>
        </div>
      </CardContent>
    </Card>
  );
};

export default ModelInfo;