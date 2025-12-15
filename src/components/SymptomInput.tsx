import { useState } from 'react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Checkbox } from '@/components/ui/checkbox';
import { Label } from '@/components/ui/label';
import { Input } from '@/components/ui/input';
import { Textarea } from '@/components/ui/textarea';
import { AlertTriangle } from 'lucide-react';

interface SymptomInputProps {
  onSubmit: (data: any) => void;
  loading: boolean;
}

const symptoms = [
  { id: 'chest_pain', label: 'Chest Pain or Discomfort', severity: 'high' },
  { id: 'shortness_breath', label: 'Shortness of Breath', severity: 'high' },
  { id: 'fatigue', label: 'Unusual Fatigue', severity: 'medium' },
  { id: 'swelling', label: 'Swelling in Legs/Ankles', severity: 'medium' },
  { id: 'irregular_heartbeat', label: 'Irregular Heartbeat', severity: 'high' },
  { id: 'dizziness', label: 'Dizziness or Lightheadedness', severity: 'medium' },
  { id: 'palpitations', label: 'Heart Palpitations', severity: 'medium' },
  { id: 'high_bp', label: 'High Blood Pressure', severity: 'high' },
  { id: 'jaw_back_pain', label: 'Jaw or Back Pain', severity: 'medium' },
  { id: 'nausea', label: 'Nausea or Vomiting', severity: 'low' },
  { id: 'fainting', label: 'Fainting Episodes', severity: 'high' },
  { id: 'cyanosis', label: 'Blue Lips or Fingernails', severity: 'high' },
  { id: 'exercise_difficulty', label: 'Difficulty During Exercise', severity: 'medium' },
];

const SymptomInput = ({ onSubmit, loading }: SymptomInputProps) => {
  const [selectedSymptoms, setSelectedSymptoms] = useState<string[]>([]);
  const [age, setAge] = useState('');
  const [weight, setWeight] = useState('');
  const [height, setHeight] = useState('');
  const [familyHistory, setFamilyHistory] = useState(false);
  const [smokingHistory, setSmokingHistory] = useState(false);
  const [additionalInfo, setAdditionalInfo] = useState('');

  const handleSymptomChange = (symptomId: string, checked: boolean) => {
    if (checked) {
      setSelectedSymptoms([...selectedSymptoms, symptomId]);
    } else {
      setSelectedSymptoms(selectedSymptoms.filter(id => id !== symptomId));
    }
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    
    const data = {
      symptoms: selectedSymptoms,
      age: parseInt(age),
      weight: parseFloat(weight),
      height: parseFloat(height),
      familyHistory,
      smokingHistory,
      additionalInfo
    };
    
    onSubmit(data);
  };

  const getSymptomColor = (severity: string) => {
    switch (severity) {
      case 'high': return 'text-destructive';
      case 'medium': return 'text-warning';
      case 'low': return 'text-muted-foreground';
      default: return 'text-foreground';
    }
  };

  return (
    <form onSubmit={handleSubmit} className="space-y-6">
      {/* Personal Information */}
      <Card>
        <CardHeader>
          <CardTitle className="text-lg">Personal Information</CardTitle>
          <CardDescription>
            Basic information to help with risk assessment
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div className="space-y-2">
              <Label htmlFor="age">Age</Label>
              <Input
                id="age"
                type="number"
                placeholder="35"
                value={age}
                onChange={(e) => setAge(e.target.value)}
                required
              />
            </div>
            <div className="space-y-2">
              <Label htmlFor="weight">Weight (kg)</Label>
              <Input
                id="weight"
                type="number"
                placeholder="70"
                value={weight}
                onChange={(e) => setWeight(e.target.value)}
              />
            </div>
            <div className="space-y-2">
              <Label htmlFor="height">Height (cm)</Label>
              <Input
                id="height"
                type="number"
                placeholder="175"
                value={height}
                onChange={(e) => setHeight(e.target.value)}
              />
            </div>
          </div>
          
          <div className="space-y-3">
            <div className="flex items-center space-x-2">
              <Checkbox
                id="family_history"
                checked={familyHistory}
                onCheckedChange={(checked) => setFamilyHistory(checked as boolean)}
              />
              <Label htmlFor="family_history">Family history of heart disease</Label>
            </div>
            
            <div className="flex items-center space-x-2">
              <Checkbox
                id="smoking_history"
                checked={smokingHistory}
                onCheckedChange={(checked) => setSmokingHistory(checked as boolean)}
              />
              <Label htmlFor="smoking_history">Current or past smoking history</Label>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Symptoms */}
      <Card>
        <CardHeader>
          <CardTitle className="text-lg">Current Symptoms</CardTitle>
          <CardDescription>
            Select all symptoms you are currently experiencing
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
            {symptoms.map((symptom) => (
              <div key={symptom.id} className="flex items-center space-x-2">
                <Checkbox
                  id={symptom.id}
                  checked={selectedSymptoms.includes(symptom.id)}
                  onCheckedChange={(checked) => handleSymptomChange(symptom.id, checked as boolean)}
                />
                <Label 
                  htmlFor={symptom.id} 
                  className={`flex-1 ${getSymptomColor(symptom.severity)}`}
                >
                  {symptom.label}
                  {symptom.severity === 'high' && (
                    <AlertTriangle className="h-3 w-3 inline ml-1" />
                  )}
                </Label>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>

      {/* Additional Information */}
      <Card>
        <CardHeader>
          <CardTitle className="text-lg">Additional Information</CardTitle>
          <CardDescription>
            Any other relevant medical information or concerns
          </CardDescription>
        </CardHeader>
        <CardContent>
          <Textarea
            placeholder="Describe any other symptoms, medications you're taking, or concerns you have..."
            value={additionalInfo}
            onChange={(e) => setAdditionalInfo(e.target.value)}
            className="min-h-[100px]"
          />
        </CardContent>
      </Card>

      <Button 
        type="submit" 
        className="w-full" 
        disabled={loading || selectedSymptoms.length === 0}
        size="lg"
      >
        {loading ? 'Analyzing...' : 'Analyze Heart Disease Risk'}
      </Button>
    </form>
  );
};

export default SymptomInput;