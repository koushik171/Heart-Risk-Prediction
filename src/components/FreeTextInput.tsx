import { useState } from 'react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Textarea } from '@/components/ui/textarea';
import { Label } from '@/components/ui/label';
import { MessageSquare, Lightbulb } from 'lucide-react';
import { Alert, AlertDescription } from '@/components/ui/alert';

interface FreeTextInputProps {
  onSubmit: (data: any) => void;
  loading: boolean;
}

const examplePrompts = [
  "I've been feeling chest tightness and fatigue for the past week, especially when climbing stairs.",
  "I experience shortness of breath and my heart sometimes races, particularly after light exercise.",
  "I have pain in my left arm and jaw, along with some nausea and dizziness.",
  "My legs are swollen and I feel tired all the time. I also have high blood pressure."
];

const FreeTextInput = ({ onSubmit, loading }: FreeTextInputProps) => {
  const [symptoms, setSymptoms] = useState('');

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    
    if (symptoms.trim().length < 10) {
      return;
    }
    
    const data = {
      description: symptoms.trim()
    };
    
    onSubmit(data);
  };

  const useExample = (example: string) => {
    setSymptoms(example);
  };

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center space-x-2">
          <MessageSquare className="h-5 w-5" />
          <span>Describe Your Symptoms</span>
        </CardTitle>
        <CardDescription>
          Tell us about your symptoms in your own words. Be as detailed as possible about what you're experiencing.
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        <Alert>
          <Lightbulb className="h-4 w-4" />
          <AlertDescription>
            Include details like: when symptoms occur, their intensity, duration, and any triggers you've noticed.
          </AlertDescription>
        </Alert>

        <form onSubmit={handleSubmit} className="space-y-4">
          <div className="space-y-2">
            <Label htmlFor="symptoms">Your Symptoms</Label>
            <Textarea
              id="symptoms"
              placeholder="Describe your symptoms here... For example: 'I've been experiencing chest pain that radiates to my left arm, especially when I walk upstairs. I also feel short of breath and sometimes dizzy.'"
              value={symptoms}
              onChange={(e) => setSymptoms(e.target.value)}
              className="min-h-[150px]"
              required
            />
            <p className="text-sm text-muted-foreground">
              {symptoms.length}/500 characters
            </p>
          </div>

          <Button 
            type="submit" 
            className="w-full" 
            disabled={loading || symptoms.trim().length < 10}
            size="lg"
          >
            {loading ? 'Analyzing...' : 'Analyze My Symptoms'}
          </Button>
        </form>

        <div className="space-y-3">
          <Label className="text-sm font-medium">Example descriptions:</Label>
          <div className="space-y-2">
            {examplePrompts.map((example, index) => (
              <div
                key={index}
                className="p-3 bg-muted/50 rounded-lg cursor-pointer hover:bg-muted transition-colors"
                onClick={() => useExample(example)}
              >
                <p className="text-sm text-muted-foreground italic">
                  "{example}"
                </p>
              </div>
            ))}
          </div>
        </div>
      </CardContent>
    </Card>
  );
};

export default FreeTextInput;