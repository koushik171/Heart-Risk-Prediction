import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Progress } from '@/components/ui/progress';
import { Heart, AlertTriangle, CheckCircle, Shield } from 'lucide-react';

interface RiskIndicatorDashboardProps {
  riskLevel: string;
  riskPercentage: number;
}

const RiskIndicatorDashboard = ({ riskLevel, riskPercentage }: RiskIndicatorDashboardProps) => {
  const getRiskConfig = (level: string) => {
    switch (level.toLowerCase()) {
      case 'low':
        return {
          color: 'bg-green-500',
          lightColor: 'bg-green-100',
          borderColor: 'border-green-300',
          textColor: 'text-green-800',
          icon: CheckCircle,
          message: 'Your heart health indicators suggest low risk. Continue maintaining healthy habits.',
          recommendations: [
            'Maintain regular exercise routine',
            'Continue healthy diet',
            'Regular health check-ups',
            'Avoid smoking and excessive alcohol'
          ]
        };
      case 'medium':
        return {
          color: 'bg-yellow-500',
          lightColor: 'bg-yellow-100',
          borderColor: 'border-yellow-300',
          textColor: 'text-yellow-800',
          icon: AlertTriangle,
          message: 'Some risk factors detected. Consider lifestyle modifications and consult healthcare provider.',
          recommendations: [
            'Increase physical activity',
            'Improve diet quality',
            'Monitor blood pressure regularly',
            'Schedule medical consultation'
          ]
        };
      case 'high':
        return {
          color: 'bg-red-500',
          lightColor: 'bg-red-100',
          borderColor: 'border-red-300',
          textColor: 'text-red-800',
          icon: Heart,
          message: 'Multiple risk factors identified. Immediate medical attention recommended.',
          recommendations: [
            'Consult cardiologist immediately',
            'Consider medication if prescribed',
            'Implement strict lifestyle changes',
            'Regular monitoring required'
          ]
        };
      default:
        return {
          color: 'bg-gray-500',
          lightColor: 'bg-gray-100',
          borderColor: 'border-gray-300',
          textColor: 'text-gray-800',
          icon: Shield,
          message: 'Risk assessment completed.',
          recommendations: []
        };
    }
  };

  const config = getRiskConfig(riskLevel);
  const IconComponent = config.icon;

  return (
    <div className="space-y-6">
      {/* Main Risk Indicator */}
      <Card className={`border-2 ${config.borderColor} ${config.lightColor}`}>
        <CardHeader className="text-center pb-2">
          <div className="flex justify-center mb-4">
            <div className={`p-4 rounded-full ${config.color}`}>
              <IconComponent className="h-8 w-8 text-white" />
            </div>
          </div>
          <CardTitle className={`text-2xl ${config.textColor}`}>
            {riskLevel.toUpperCase()} RISK DETECTED
          </CardTitle>
          <CardDescription className={config.textColor}>
            Risk Score: {riskPercentage.toFixed(1)}%
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          {/* Risk Meter */}
          <div className="space-y-2">
            <div className="flex justify-between text-sm font-medium">
              <span className="text-green-600">Low</span>
              <span className="text-yellow-600">Medium</span>
              <span className="text-red-600">High</span>
            </div>
            <div className="relative">
              <Progress value={riskPercentage} className="h-4" />
              <div className="absolute inset-0 flex">
                <div className="w-1/3 border-r border-white"></div>
                <div className="w-1/3 border-r border-white"></div>
                <div className="w-1/3"></div>
              </div>
            </div>
            <div className="flex justify-between text-xs text-gray-500">
              <span>0-30%</span>
              <span>31-70%</span>
              <span>71-100%</span>
            </div>
          </div>

          {/* Risk Message */}
          <div className={`p-4 rounded-lg ${config.lightColor} border ${config.borderColor}`}>
            <p className={`text-sm ${config.textColor} font-medium`}>
              {config.message}
            </p>
          </div>
        </CardContent>
      </Card>

      {/* Risk Level Legend */}
      <Card>
        <CardHeader>
          <CardTitle className="text-lg">Risk Level Guide</CardTitle>
          <CardDescription>
            Understanding your heart disease risk assessment
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            {/* Low Risk */}
            <div className="flex items-center space-x-3 p-3 bg-green-50 rounded-lg border border-green-200">
              <div className="p-2 bg-green-500 rounded-full">
                <CheckCircle className="h-4 w-4 text-white" />
              </div>
              <div>
                <div className="font-semibold text-green-800">Low Risk</div>
                <div className="text-sm text-green-600">0-30%</div>
              </div>
            </div>

            {/* Medium Risk */}
            <div className="flex items-center space-x-3 p-3 bg-yellow-50 rounded-lg border border-yellow-200">
              <div className="p-2 bg-yellow-500 rounded-full">
                <AlertTriangle className="h-4 w-4 text-white" />
              </div>
              <div>
                <div className="font-semibold text-yellow-800">Medium Risk</div>
                <div className="text-sm text-yellow-600">31-70%</div>
              </div>
            </div>

            {/* High Risk */}
            <div className="flex items-center space-x-3 p-3 bg-red-50 rounded-lg border border-red-200">
              <div className="p-2 bg-red-500 rounded-full">
                <Heart className="h-4 w-4 text-white" />
              </div>
              <div>
                <div className="font-semibold text-red-800">High Risk</div>
                <div className="text-sm text-red-600">71-100%</div>
              </div>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Recommendations */}
      {config.recommendations.length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle className="text-lg">Recommended Actions</CardTitle>
            <CardDescription>
              Based on your {riskLevel.toLowerCase()} risk assessment
            </CardDescription>
          </CardHeader>
          <CardContent>
            <ul className="space-y-2">
              {config.recommendations.map((recommendation, index) => (
                <li key={index} className="flex items-start space-x-2">
                  <div className={`p-1 rounded-full ${config.color} mt-1`}>
                    <div className="w-2 h-2 bg-white rounded-full"></div>
                  </div>
                  <span className="text-sm">{recommendation}</span>
                </li>
              ))}
            </ul>
          </CardContent>
        </Card>
      )}
    </div>
  );
};

export default RiskIndicatorDashboard;