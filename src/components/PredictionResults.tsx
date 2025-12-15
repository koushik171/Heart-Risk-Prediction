import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Progress } from '@/components/ui/progress';
import { Separator } from '@/components/ui/separator';
import { Heart, TrendingUp, AlertTriangle, CheckCircle, FileText, Mail } from 'lucide-react';
import { Alert, AlertDescription } from '@/components/ui/alert';
import RiskIndicatorDashboard from './RiskIndicatorDashboard';

interface PredictionResultsProps {
  data: any;
}

const PredictionResults = ({ data }: PredictionResultsProps) => {
  const getRiskColor = (level: string) => {
    switch (level.toLowerCase()) {
      case 'low': return 'bg-green-500 text-white border-green-500';
      case 'medium': return 'bg-yellow-500 text-white border-yellow-500';
      case 'high': return 'bg-red-500 text-white border-red-500';
      default: return 'bg-gray-500 text-white border-gray-500';
    }
  };

  const getRiskIndicatorColor = (level: string) => {
    switch (level.toLowerCase()) {
      case 'low': return 'bg-green-100 border-green-300';
      case 'medium': return 'bg-yellow-100 border-yellow-300';
      case 'high': return 'bg-red-100 border-red-300';
      default: return 'bg-gray-100 border-gray-300';
    }
  };

  const getProgressColor = (level: string) => {
    switch (level.toLowerCase()) {
      case 'low': return '[&>div]:bg-green-500';
      case 'medium': return '[&>div]:bg-yellow-500';
      case 'high': return '[&>div]:bg-red-500';
      default: return '[&>div]:bg-gray-500';
    }
  };

  const getRiskIcon = (level: string) => {
    switch (level.toLowerCase()) {
      case 'low': return <CheckCircle className="h-4 w-4" />;
      case 'medium': return <AlertTriangle className="h-4 w-4" />;
      case 'high': return <Heart className="h-4 w-4" />;
      default: return null;
    }
  };

  const handleGenerateReport = () => {
    // Mock PDF generation
    console.log('Generating PDF report...');
  };

  const handleEmailReport = () => {
    // Mock email functionality
    console.log('Emailing report...');
  };

  return (
    <div className="space-y-6">
      {/* Enhanced Risk Assessment Dashboard */}
      <RiskIndicatorDashboard 
        riskLevel={data.riskLevel} 
        riskPercentage={data.riskPercentage} 
      />
      
      <Alert>
        <AlertTriangle className="h-4 w-4" />
        <AlertDescription>
          <strong>Medical Disclaimer:</strong> This is an AI prediction for educational purposes only. 
          Please consult with a healthcare professional for proper medical advice and diagnosis.
        </AlertDescription>
      </Alert>

      {/* SHAP Explanation */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center space-x-2">
            <TrendingUp className="h-5 w-5 text-blue-600" />
            <span>Risk Factor Analysis</span>
          </CardTitle>
          <CardDescription>
            Key factors contributing to your risk assessment
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            {data.shapExplanation.topFeatures.map((feature: any, index: number) => {
              const impactLevel = feature.importance > 0.3 ? 'high' : feature.importance > 0.2 ? 'medium' : 'low';
              const impactColor = impactLevel === 'high' ? 'bg-red-100 border-red-300' : 
                                 impactLevel === 'medium' ? 'bg-yellow-100 border-yellow-300' : 
                                 'bg-green-100 border-green-300';
              
              return (
                <div key={index} className={`p-4 rounded-lg border ${impactColor}`}>
                  <div className="flex justify-between items-center mb-2">
                    <span className="font-semibold text-gray-800">{feature.feature}</span>
                    <span className="text-sm font-medium px-2 py-1 rounded bg-white border">
                      {feature.value}
                    </span>
                  </div>
                  <div className="flex items-center space-x-2">
                    <Progress 
                      value={feature.importance * 100} 
                      className={`flex-1 h-2 ${impactLevel === 'high' ? '[&>div]:bg-red-500' : 
                                                impactLevel === 'medium' ? '[&>div]:bg-yellow-500' : 
                                                '[&>div]:bg-green-500'}`} 
                    />
                    <span className="text-sm font-medium text-gray-600">
                      {(feature.importance * 100).toFixed(1)}% impact
                    </span>
                  </div>
                </div>
              );
            })}
          </div>
        </CardContent>
      </Card>

      {/* Predicted Condition */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center space-x-2">
            <Heart className="h-5 w-5 text-red-600" />
            <span>Assessment Focus</span>
          </CardTitle>
          <CardDescription>
            Primary cardiovascular condition assessed
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className={`p-6 rounded-lg border-2 ${
            data.riskLevel.toLowerCase() === 'high' ? 'bg-red-50 border-red-200' :
            data.riskLevel.toLowerCase() === 'medium' ? 'bg-yellow-50 border-yellow-200' :
            'bg-green-50 border-green-200'
          }`}>
            <h3 className="font-bold text-xl text-gray-800 mb-3">{data.predictedCondition}</h3>
            <p className="text-gray-700 leading-relaxed">
              Coronary Artery Disease (CAD) is the most common type of heart disease. It occurs when 
              the arteries that supply blood to your heart muscle become hardened and narrowed due to 
              plaque buildup. Early detection and lifestyle modifications can significantly reduce risk.
            </p>
            <div className="mt-4 p-3 bg-white rounded border">
              <p className="text-sm font-medium text-gray-600">
                <strong>Note:</strong> This assessment is based on risk factors and symptoms. 
                Definitive diagnosis requires medical evaluation and testing.
              </p>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Diet Recommendations */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center space-x-2">
            <Heart className="h-5 w-5 text-green-600" />
            <span>Heart-Healthy Diet Plan</span>
          </CardTitle>
          <CardDescription>
            Personalized nutrition recommendations for your risk level
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="bg-green-50 border border-green-200 rounded-lg p-4">
            <h4 className="font-semibold text-green-800 mb-3">Recommended Dietary Changes</h4>
            <ul className="space-y-3">
              {data.dietRecommendations.map((recommendation: string, index: number) => (
                <li key={index} className="flex items-start space-x-3">
                  <div className="p-1 bg-green-500 rounded-full mt-1">
                    <CheckCircle className="h-3 w-3 text-white" />
                  </div>
                  <span className="text-sm text-green-800 leading-relaxed">{recommendation}</span>
                </li>
              ))}
            </ul>
          </div>
        </CardContent>
      </Card>

      {/* Medication Advice */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center space-x-2">
            <AlertTriangle className="h-5 w-5 text-blue-600" />
            <span>Medical Consultation Guidance</span>
          </CardTitle>
          <CardDescription>
            Important information to discuss with your healthcare provider
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className={`p-6 rounded-lg border-2 ${
            data.riskLevel.toLowerCase() === 'high' ? 'bg-red-50 border-red-200' :
            data.riskLevel.toLowerCase() === 'medium' ? 'bg-yellow-50 border-yellow-200' :
            'bg-blue-50 border-blue-200'
          }`}>
            <div className="mb-4">
              <h4 className={`font-semibold mb-2 ${
                data.riskLevel.toLowerCase() === 'high' ? 'text-red-800' :
                data.riskLevel.toLowerCase() === 'medium' ? 'text-yellow-800' :
                'text-blue-800'
              }`}>
                {data.riskLevel === 'High' ? 'Urgent Medical Attention Required' :
                 data.riskLevel === 'Medium' ? 'Schedule Medical Consultation' :
                 'Routine Health Monitoring'}
              </h4>
              <p className={`text-sm leading-relaxed ${
                data.riskLevel.toLowerCase() === 'high' ? 'text-red-700' :
                data.riskLevel.toLowerCase() === 'medium' ? 'text-yellow-700' :
                'text-blue-700'
              }`}>
                {data.medicationAdvice}
              </p>
            </div>
            
            <Alert className={`${
              data.riskLevel.toLowerCase() === 'high' ? 'border-red-300 bg-red-100' :
              data.riskLevel.toLowerCase() === 'medium' ? 'border-yellow-300 bg-yellow-100' :
              'border-blue-300 bg-blue-100'
            }`}>
              <AlertTriangle className="h-4 w-4" />
              <AlertDescription className="text-xs font-medium">
                <strong>Critical Reminder:</strong> This assessment is for informational purposes only. 
                Always consult qualified healthcare professionals before making any medical decisions. 
                Never start, stop, or change medications without proper medical supervision.
              </AlertDescription>
            </Alert>
          </div>
        </CardContent>
      </Card>

      {/* Actions */}
      <Card>
        <CardHeader>
          <CardTitle className="text-lg">Next Steps</CardTitle>
          <CardDescription>
            Save and share your risk assessment results
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <Button 
              onClick={handleGenerateReport} 
              className="h-12"
            >
              <FileText className="h-4 w-4 mr-2" />
              Generate Detailed Report
            </Button>
            <Button 
              variant="outline" 
              onClick={handleEmailReport} 
              className="h-12"
            >
              <Mail className="h-4 w-4 mr-2" />
              Email to Doctor
            </Button>
          </div>
          
          {data.riskLevel.toLowerCase() === 'high' && (
            <div className="mt-4 p-4 bg-red-100 border border-red-300 rounded-lg">
              <p className="text-sm font-semibold text-red-800 mb-2">
                🚨 High Risk Detected - Immediate Action Required
              </p>
              <p className="text-xs text-red-700">
                Please schedule an appointment with a cardiologist as soon as possible. 
                Consider this assessment as urgent and do not delay seeking medical attention.
              </p>
            </div>
          )}
          
          {data.riskLevel.toLowerCase() === 'medium' && (
            <div className="mt-4 p-4 bg-yellow-100 border border-yellow-300 rounded-lg">
              <p className="text-sm font-semibold text-yellow-800 mb-2">
                ⚠️ Moderate Risk - Medical Consultation Recommended
              </p>
              <p className="text-xs text-yellow-700">
                Schedule a consultation with your healthcare provider within the next few weeks 
                to discuss these findings and develop a prevention plan.
              </p>
            </div>
          )}
          
          {data.riskLevel.toLowerCase() === 'low' && (
            <div className="mt-4 p-4 bg-green-100 border border-green-300 rounded-lg">
              <p className="text-sm font-semibold text-green-800 mb-2">
                ✅ Low Risk - Continue Healthy Habits
              </p>
              <p className="text-xs text-green-700">
                Great job! Continue your current healthy lifestyle and maintain regular 
                check-ups with your healthcare provider.
              </p>
            </div>
          )}
        </CardContent>
      </Card>

      <Separator />
      
      <div className="text-center text-sm text-muted-foreground">
        <p>Assessment completed on {new Date(data.createdAt).toLocaleDateString()}</p>
      </div>
    </div>
  );
};

export default PredictionResults;