import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { History, Eye, FileText, Calendar } from 'lucide-react';

const mockHistory = [
  {
    id: '1',
    date: '2024-01-15T10:30:00Z',
    inputType: 'structured',
    riskLevel: 'Low',
    riskPercentage: 15.2,
    symptoms: ['fatigue', 'mild_chest_discomfort'],
    predictedCondition: 'No significant risk detected'
  },
  {
    id: '2',
    date: '2024-01-10T14:45:00Z',
    inputType: 'freetext',
    riskLevel: 'Medium',
    riskPercentage: 42.8,
    symptoms: 'chest pain and shortness of breath',
    predictedCondition: 'Possible Angina'
  },
  {
    id: '3',
    date: '2024-01-05T09:15:00Z',
    inputType: 'structured',
    riskLevel: 'High',
    riskPercentage: 78.3,
    symptoms: ['chest_pain', 'shortness_breath', 'irregular_heartbeat'],
    predictedCondition: 'Coronary Artery Disease'
  }
];

const PredictionHistory = () => {
  const getRiskColor = (level: string) => {
    switch (level.toLowerCase()) {
      case 'low': return 'secondary';
      case 'medium': return 'outline';
      case 'high': return 'destructive';
      default: return 'secondary';
    }
  };

  const formatDate = (dateString: string) => {
    return new Date(dateString).toLocaleDateString('en-US', {
      year: 'numeric',
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit'
    });
  };

  const handleViewDetails = (id: string) => {
    console.log('Viewing details for prediction:', id);
  };

  const handleDownloadReport = (id: string) => {
    console.log('Downloading report for prediction:', id);
  };

  return (
    <div className="space-y-6">
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center space-x-2">
            <History className="h-6 w-6 text-primary" />
            <span>Prediction History</span>
          </CardTitle>
          <CardDescription>
            View your past heart disease risk assessments and track changes over time
          </CardDescription>
        </CardHeader>
      </Card>

      {mockHistory.length === 0 ? (
        <Card>
          <CardContent className="flex items-center justify-center h-64">
            <div className="text-center space-y-3">
              <History className="h-12 w-12 text-muted-foreground mx-auto" />
              <div>
                <p className="text-muted-foreground">No assessments yet</p>
                <p className="text-sm text-muted-foreground">
                  Complete your first heart disease risk assessment to see your history here
                </p>
              </div>
            </div>
          </CardContent>
        </Card>
      ) : (
        <div className="space-y-4">
          {mockHistory.map((prediction) => (
            <Card key={prediction.id} className="hover:shadow-md transition-shadow">
              <CardContent className="p-6">
                <div className="flex items-center justify-between mb-4">
                  <div className="flex items-center space-x-3">
                    <Calendar className="h-5 w-5 text-muted-foreground" />
                    <div>
                      <p className="font-medium">{formatDate(prediction.date)}</p>
                      <p className="text-sm text-muted-foreground capitalize">
                        {prediction.inputType === 'freetext' ? 'Free Text' : 'Guided Form'} Assessment
                      </p>
                    </div>
                  </div>
                  <div className="flex items-center space-x-2">
                    <Badge variant={getRiskColor(prediction.riskLevel)}>
                      {prediction.riskLevel} Risk
                    </Badge>
                    <span className="text-lg font-semibold">
                      {prediction.riskPercentage.toFixed(1)}%
                    </span>
                  </div>
                </div>

                <div className="space-y-3">
                  <div>
                    <p className="text-sm font-medium text-muted-foreground">Predicted Condition</p>
                    <p className="text-sm">{prediction.predictedCondition}</p>
                  </div>

                  <div>
                    <p className="text-sm font-medium text-muted-foreground">Symptoms/Input</p>
                    <p className="text-sm">
                      {Array.isArray(prediction.symptoms) 
                        ? prediction.symptoms.join(', ').replace(/_/g, ' ')
                        : prediction.symptoms}
                    </p>
                  </div>
                </div>

                <div className="flex justify-end space-x-2 mt-4 pt-4 border-t">
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={() => handleViewDetails(prediction.id)}
                  >
                    <Eye className="h-4 w-4 mr-2" />
                    View Details
                  </Button>
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={() => handleDownloadReport(prediction.id)}
                  >
                    <FileText className="h-4 w-4 mr-2" />
                    Download Report
                  </Button>
                </div>
              </CardContent>
            </Card>
          ))}
        </div>
      )}

      {mockHistory.length > 0 && (
        <Card>
          <CardContent className="p-6">
            <div className="text-center space-y-2">
              <h3 className="font-medium">Risk Trend Analysis</h3>
              <p className="text-sm text-muted-foreground">
                Your latest assessment shows a {mockHistory[0].riskLevel.toLowerCase()} risk level. 
                {mockHistory[0].riskLevel === 'High' && 
                  " We recommend consulting with a healthcare provider for a thorough evaluation."
                }
                {mockHistory[0].riskLevel === 'Medium' && 
                  " Consider lifestyle modifications and regular monitoring."
                }
                {mockHistory[0].riskLevel === 'Low' && 
                  " Keep up the good work with your heart-healthy lifestyle!"
                }
              </p>
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  );
};

export default PredictionHistory;