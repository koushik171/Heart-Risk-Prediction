import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Accordion, AccordionContent, AccordionItem, AccordionTrigger } from '@/components/ui/accordion';
import { Heart, Activity, Zap, AlertTriangle, Shield } from 'lucide-react';

const heartDiseases = [
  {
    id: 'coronary',
    title: 'Coronary Artery Disease',
    icon: Heart,
    description: 'The most common type of heart disease, caused by plaque buildup in arteries.',
    symptoms: ['Chest pain', 'Shortness of breath', 'Fatigue', 'Heart palpitations'],
    causes: ['High cholesterol', 'High blood pressure', 'Smoking', 'Diabetes', 'Family history'],
    prevention: ['Regular exercise', 'Healthy diet', 'No smoking', 'Stress management', 'Regular checkups']
  },
  {
    id: 'arrhythmia',
    title: 'Arrhythmia',
    icon: Zap,
    description: 'Irregular heartbeat that can be too fast, too slow, or erratic.',
    symptoms: ['Palpitations', 'Dizziness', 'Fainting', 'Chest discomfort', 'Shortness of breath'],
    causes: ['Heart disease', 'Stress', 'Caffeine', 'Alcohol', 'Certain medications'],
    prevention: ['Limit caffeine', 'Manage stress', 'Avoid smoking', 'Moderate alcohol', 'Regular sleep']
  },
  {
    id: 'heart_failure',
    title: 'Heart Failure',
    icon: Activity,
    description: 'Condition where the heart cannot pump blood effectively to meet the body\'s needs.',
    symptoms: ['Swelling in legs/ankles', 'Fatigue', 'Shortness of breath', 'Rapid weight gain'],
    causes: ['Coronary artery disease', 'High blood pressure', 'Previous heart attack', 'Diabetes'],
    prevention: ['Control blood pressure', 'Manage diabetes', 'Healthy weight', 'Limit sodium', 'Regular exercise']
  },
  {
    id: 'valve_disease',
    title: 'Heart Valve Disease',
    icon: Shield,
    description: 'Problems with one or more of the heart\'s four valves.',
    symptoms: ['Fatigue', 'Shortness of breath', 'Swelling', 'Chest pain', 'Irregular heartbeat'],
    causes: ['Age-related changes', 'Rheumatic fever', 'Infections', 'Birth defects'],
    prevention: ['Prevent infections', 'Treat strep throat', 'Good dental hygiene', 'Regular checkups']
  }
];

const riskFactors = [
  {
    title: 'Modifiable Risk Factors',
    description: 'Factors you can control through lifestyle changes',
    factors: [
      'High blood pressure',
      'High cholesterol',
      'Smoking',
      'Physical inactivity',
      'Obesity',
      'Diabetes',
      'Unhealthy diet',
      'Excessive alcohol',
      'Stress'
    ]
  },
  {
    title: 'Non-Modifiable Risk Factors',
    description: 'Factors you cannot change but should be aware of',
    factors: [
      'Age (men >45, women >55)',
      'Gender (men at higher risk)',
      'Family history',
      'Race/ethnicity',
      'Previous heart attack',
      'Previous stroke'
    ]
  }
];

const EducationPanel = () => {
  return (
    <div className="space-y-6">
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center space-x-2">
            <Heart className="h-6 w-6 text-primary" />
            <span>Heart Disease Education</span>
          </CardTitle>
          <CardDescription>
            Learn about different types of cardiovascular conditions, their symptoms, and prevention strategies
          </CardDescription>
        </CardHeader>
      </Card>

      {/* Types of Heart Disease */}
      <Card>
        <CardHeader>
          <CardTitle>Types of Heart Disease</CardTitle>
          <CardDescription>
            Understanding different cardiovascular conditions and their characteristics
          </CardDescription>
        </CardHeader>
        <CardContent>
          <Accordion type="single" collapsible className="w-full">
            {heartDiseases.map((disease) => {
              const IconComponent = disease.icon;
              return (
                <AccordionItem key={disease.id} value={disease.id}>
                  <AccordionTrigger className="text-left">
                    <div className="flex items-center space-x-3">
                      <IconComponent className="h-5 w-5 text-primary" />
                      <div>
                        <div className="font-medium">{disease.title}</div>
                        <div className="text-sm text-muted-foreground">{disease.description}</div>
                      </div>
                    </div>
                  </AccordionTrigger>
                  <AccordionContent className="space-y-4">
                    <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                      <div>
                        <h4 className="font-medium text-sm mb-2 text-destructive">Common Symptoms</h4>
                        <ul className="text-sm space-y-1">
                          {disease.symptoms.map((symptom, index) => (
                            <li key={index} className="flex items-center space-x-2">
                              <AlertTriangle className="h-3 w-3 text-destructive" />
                              <span>{symptom}</span>
                            </li>
                          ))}
                        </ul>
                      </div>
                      <div>
                        <h4 className="font-medium text-sm mb-2 text-warning">Risk Factors</h4>
                        <ul className="text-sm space-y-1">
                          {disease.causes.map((cause, index) => (
                            <li key={index} className="flex items-center space-x-2">
                              <AlertTriangle className="h-3 w-3 text-warning" />
                              <span>{cause}</span>
                            </li>
                          ))}
                        </ul>
                      </div>
                      <div>
                        <h4 className="font-medium text-sm mb-2 text-success">Prevention</h4>
                        <ul className="text-sm space-y-1">
                          {disease.prevention.map((prevention, index) => (
                            <li key={index} className="flex items-center space-x-2">
                              <Shield className="h-3 w-3 text-success" />
                              <span>{prevention}</span>
                            </li>
                          ))}
                        </ul>
                      </div>
                    </div>
                  </AccordionContent>
                </AccordionItem>
              );
            })}
          </Accordion>
        </CardContent>
      </Card>

      {/* Risk Factors */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {riskFactors.map((category, index) => (
          <Card key={index}>
            <CardHeader>
              <CardTitle className="text-lg">{category.title}</CardTitle>
              <CardDescription>{category.description}</CardDescription>
            </CardHeader>
            <CardContent>
              <ul className="space-y-2">
                {category.factors.map((factor, factorIndex) => (
                  <li key={factorIndex} className="flex items-center space-x-2">
                    <div className={`h-2 w-2 rounded-full ${index === 0 ? 'bg-warning' : 'bg-muted-foreground'}`} />
                    <span className="text-sm">{factor}</span>
                  </li>
                ))}
              </ul>
            </CardContent>
          </Card>
        ))}
      </div>

      {/* Quick Tips */}
      <Card>
        <CardHeader>
          <CardTitle>Heart-Healthy Living Tips</CardTitle>
          <CardDescription>
            Simple daily habits that can significantly reduce your heart disease risk
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {[
              { title: 'Exercise Regularly', tip: '150 minutes moderate activity per week' },
              { title: 'Eat Heart-Healthy', tip: 'Focus on fruits, vegetables, whole grains' },
              { title: 'Manage Stress', tip: 'Try meditation, yoga, or deep breathing' },
              { title: 'Get Quality Sleep', tip: '7-9 hours of sleep each night' },
              { title: 'Quit Smoking', tip: 'Seek support to quit tobacco use' },
              { title: 'Limit Alcohol', tip: 'No more than 1-2 drinks per day' },
            ].map((tip, index) => (
              <div key={index} className="p-4 bg-muted/50 rounded-lg">
                <h4 className="font-medium mb-1">{tip.title}</h4>
                <p className="text-sm text-muted-foreground">{tip.tip}</p>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>
    </div>
  );
};

export default EducationPanel;